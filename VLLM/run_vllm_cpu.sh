#!/bin/bash
#SBATCH --job-name=vllm_cpu
#SBATCH --partition=compute      # m01-m48: 2x AMD EPYC 7763, 128 cores, 512 GB RAM
#SBATCH --nodes=1                # specs: https://doc25.aris.grnet.gr/system/hardware/
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20       # our choice; vLLM pins OMP threads to ONE NUMA node (16 cores)
#SBATCH --mem=56G                # 512 GB nominal (SLURM RealMemory=507904 MB); 56G kept on purpose
#SBATCH --time=02:00:00          # Set walltime limit
#SBATCH --output=OUTPUTS/vllm_cpu_%j.out
#SBATCH --error=OUTPUTS/vllm_cpu_%j.err
#SBATCH --account=pr020006_thin    # Replace with your system project

#export WORKDIR=/work/pr020/pskarvel  # uncomment if WORKDIR is not set in the environment
module load gnu/13.3.0
module load python/3.12.9
source ~/MITOS-QA/VLLM/vllm_cpu_env/bin/activate

# 2. Performance tuning for ARIS thin nodes (AMD EPYC 7763 "Milan", 8 NUMA nodes).
#    IMPORTANT: do NOT export OMP_NUM_THREADS (nor MKL_/OPENBLAS_ thread vars) here.
#    This vLLM manages the OpenMP environment itself: it reserves 1 CPU, pins the
#    OMP threads to the NUMA node of the allocated cores and drops torch intra-op
#    threads to 1 (avoids spin-wait contention). An externally set OMP_NUM_THREADS
#    is detected and *disables* that last optimization (see logs: "Multi-threaded
#    torch CPU ops during serving can degrade performance...").
# (VLLM_CPU_KVCACHE_SPACE default is chosen in section 4 - it depends on VLLM_TP:
#  with TP>1 the KV cache is PER RANK!)
export VLLM_CPU_OMP_THREADS_BIND=auto      # bind OMP threads to one NUMA node

# 2a. Fix vLLM's "auto" thread-binding on ARIS: it picks the FIRST allowed NUMA
#     node, but SLURM here leaves Mems_allowed_list=0-7 even when the allocated
#     CPUs sit on another node -> vLLM selects NUMA node 0, finds zero allocated
#     cores and logs:
#       "Selected CPU core number (0) should be greater than reserved CPU core
#        number (1)"  +  "core ids=[]"   -> no thread pinning at all.
#     Tell vLLM which NUMA nodes this job actually lives on:
export CPU_VISIBLE_MEMORY_NODES=$(python - <<'PY'
import json, os, subprocess
allowed = {int(c) for c in os.sched_getaffinity(0)}
cpus = json.loads(subprocess.check_output(
    ["lscpu", "--json", "--extended=CPU,NODE", "--online"], text=True))["cpus"]
nodes = sorted({int(e["node"]) for e in cpus
                if str(e["node"]).isdigit() and int(e["cpu"]) in allowed})
print(",".join(map(str, nodes)))
PY
)
echo "CPU_VISIBLE_MEMORY_NODES=$CPU_VISIBLE_MEMORY_NODES (allocation: $(taskset -pc $$ | sed 's/.*: //'))"

# 2b. Preload tcmalloc + Intel OpenMP. Without them vLLM logs
#     "libtcmalloc/libiomp is not found in LD_PRELOAD" and runs slower
#     (documented perf feature of the CPU backend).
#       - tcmalloc: compute nodes have NO libtcmalloc (stripped image).
#         It was copied from login05 into ~/toolchain/preload by
#         TOOLS/setup_preload_libs.sh - re-run that script if the file is missing.
#       - libiomp5 : ships inside the venv (intel-openmp, a vLLM CPU dependency).
export LD_PRELOAD="$HOME/toolchain/preload/libtcmalloc_minimal.so.4:$HOME/MITOS-QA/VLLM/vllm_cpu_env/lib/libiomp5.so"

# 3. Environment variables for vLLM configuration
#    (VLLM_ENABLE_V1_MULTIPROCESSING=0 was removed: with 0, vLLM uses the "uni"
#     executor, which cannot set up the OMP thread environment correctly - the
#     thread-binding info never appears in the logs and torch stays at 20 threads)

# 3b. Fix for ARIS compute nodes: they mount an *empty* /usr/include (no glibc
#     development headers), so torch.compile / inductor JIT cannot find
#     <features.h>, <stdlib.h>, Python.h, or the crt*/lib*.so link files.
#     ~/toolchain contains a copy of the needed files from the login node:
#       - bin/g++ and bin/gcc wrappers add "-idirafter ..." so the copied
#         headers are searched exactly where /usr/include normally sits
#         (important: #include_next chains must find them)
#       - glibc/usr/include : rsync copy of /usr/include from the login node
#       - glibc/usr/lib64   : crti.o, crtn.o, libc.so, libm.so, stub -l libs
#     If this ever breaks again, fallback: add "--enforce-eager" to vllm serve
#     below (starts without any JIT compile, but CPU inference is slower).
export PATH=$HOME/toolchain/bin:$PATH
export LIBRARY_PATH=$HOME/toolchain/glibc/usr/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}

# 4. Launch the vLLM OpenAI-compatible API server
#
# dtype choice (measured 2026-10-13, idle node m35, 20 CPUs = one NUMA node,
# aggregate tok/s at client concurrency b; see TOOLS/bench_scale.sh):
#
#   MODE A - interactive chat (default): --dtype bfloat16
#     b=1: 2.7 tok/s   b=8: 8.7   b=32: 8.7 (PLATEAUS - torch fallback path)
#   MODE B - bulk QA / evaluation: --dtype float32 + concurrent requests
#     b=1: 1.4 tok/s   b=8: 10.7  b=32: 35.2  b=64: 47.4  b=128: 49.4 (~saturates)
#
# Why: only fp32 engages the optimized oneDNN GEMMs on these AVX2-only Zen 3
# nodes ("using oneDNN onednn_mm"); fp16/bf16 always fall back to torch and
# plateau at ~8.7 tok/s aggregate. fp32 KV cache needs 32 GiB for the full
# 131072 context -> use a smaller max-model-len:
#   VLLM_DTYPE=float32 VLLM_MAX_LEN=32768 VLLM_CPU_KVCACHE_SPACE=16 sbatch run_vllm_cpu.sh
# and send requests with concurrency 32-128 (continuous batching).
# fp32 adds no quality (model is bf16-native, fp32 is a pure upcast).
dtype=${VLLM_DTYPE:-bfloat16}
max_len=${VLLM_MAX_LEN:-}                  # empty = model default (131072)
tp=${VLLM_TP:-1}                           # 8 = full node, one rank per NUMA node (see bottom)
# hide our custom knobs from vLLM's environment scanner (otherwise vLLM prints
# harmless "Unknown vLLM environment variable detected: VLLM_TP ..." warnings)
unset VLLM_DTYPE VLLM_MAX_LEN VLLM_TP 2>/dev/null || true

# KV cache is PER RANK when tp>1! Job memory must cover ~tp*15.5 GiB transient
# weight-load peak + tp*KV. The single-rank default (20 GiB x 8 ranks = 160 GiB of
# KV alone) caused a cgroup OOM kill in job 171567 - so default to 4 GiB/rank for TP>1.
if [ -z "${VLLM_CPU_KVCACHE_SPACE:-}" ]; then
    if [ "$tp" -gt 1 ]; then VLLM_CPU_KVCACHE_SPACE=4; else VLLM_CPU_KVCACHE_SPACE=20; fi
fi
export VLLM_CPU_KVCACHE_SPACE
if [ "$tp" -gt 1 ] && [ "$((tp * VLLM_CPU_KVCACHE_SPACE))" -gt 80 ]; then
    echo "WARNING: tp=$tp x KV=${VLLM_CPU_KVCACHE_SPACE}G = $((tp * VLLM_CPU_KVCACHE_SPACE)) GiB of KV cache - check --mem!"
fi
echo "serving: dtype=$dtype max_len=${max_len:-default} tp=$tp KV/rank=${VLLM_CPU_KVCACHE_SPACE}G"

vllm serve ${WORKDIR}/models/Llama-Krikri-8B-Instruct \
    --port 8000 \
    --dtype "$dtype" \
    ${max_len:+--max-model-len "$max_len"} \
    --tensor-parallel-size "$tp"

# ---- Tensor parallel across NUMA nodes - VALIDATED 2026-10-13 (job 171488, full node) ----
# Full-node test (TP=8, 128 CPUs): each rank is bound to ITS OWN NUMA node
# (log "OpenMP thread binding info": rank r -> cores 16r..16r+14) - the
# CPU_VISIBLE_MEMORY_NODES fix above makes this work automatically.
#
#   aggregate tok/s        b=1    b=8    b=32    b=64    b=128   b=256
#   single-rank bf16        2.7    8.7    8.7     -       -       -    (plateau)
#   single-rank fp32        1.4   10.7   35.2   47.4    49.4     -    (plateau)
#   TP=8 bf16               6.1   31.1   46.1   47.7    49.5     -    (plateau ~50)
#   TP=8 fp32               4.6    -     81.9  106.8   114.5   128.4  (+5%/2x batch; ceiling ~130+)
#
# Usage (env overrides pass through SLURM; --mem=160G covers the ~125 GiB transient
# weight-load peak of 8 ranks; KV auto-defaults to 4 GiB/rank when TP>1):
#   VLLM_TP=8 sbatch --cpus-per-task=128 --mem=160G run_vllm_cpu.sh
#   VLLM_TP=8 VLLM_DTYPE=float32 VLLM_MAX_LEN=16384 VLLM_CPU_KVCACHE_SPACE=8 \
#       sbatch --cpus-per-task=128 --mem=160G run_vllm_cpu.sh
# Reproduction: TOOLS/bench_tp.sh (full node) / TOOLS/bench_scale.sh (single node).
