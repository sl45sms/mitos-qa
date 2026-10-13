#!/bin/bash
#SBATCH --job-name=bench_dtype
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=56G
#SBATCH --time=00:45:00
#SBATCH --account=pr020006_thin
#SBATCH --output=OUTPUTS/bench_dtype_%j.out
#SBATCH --error=OUTPUTS/bench_dtype_%j.err
#
# A/B benchmark of the vLLM dtype configs on ARIS thin nodes (Zen 3 = no FP16 compute!)
#
# Scheduler reference (in-job):
#   job 171381: 64 tok / 63.8 s  ~= 1.0 tok/s
set -u

export WORKDIR=/work/pr020/pskarvel
module load gnu/13.3.0
module load python/3.12.9
source ~/MITOS-QA/VLLM/vllm_cpu_env/bin/activate

# Toolchain (same as run_vllm_cpu.sh section 3b)
export PATH=$HOME/toolchain/bin:$PATH
export LIBRARY_PATH=$HOME/toolchain/glibc/usr/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}

# tcmalloc + Intel OMP (same as run_vllm_cpu.sh will use)
export LD_PRELOAD="$HOME/toolchain/preload/libtcmalloc_minimal.so.4:$HOME/MITOS-QA/VLLM/vllm_cpu_env/lib/libiomp5.so"

# Let vLLM manage OMP/thread binding itself (do NOT set OMP_NUM_THREADS!)
export VLLM_CPU_OMP_THREADS_BIND=auto
export VLLM_CPU_NUM_OF_RESERVED_CPU=1

# Auto-binding fix: vLLM picks the FIRST allowed NUMA node, but ARIS SLURM keeps
# Mems_allowed_list=0-7 even when the allocated CPUs are elsewhere -> without
# this, binding selects NUMA node 0, finds no cores (core ids=[]) and disables
# thread pinning. Point vLLM at the NUMA nodes of the actual allocation.
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

MODEL=${WORKDIR}/models/Llama-Krikri-8B-Instruct
PROMPT='Γράψε μια σύντομη παράγραφο για τη θάλασσα.'

# The python phases run via TOOLS/bench_worker.py (a real file!): vLLM's engine
# core is a multiprocessing "spawn" child which must re-open the parent __main__
# file - a stdin heredoc fails with "FileNotFoundError ... '<stdin>'".
# cd to $HOME so nothing in the cwd can shadow the editable vllm install.
cd "$HOME"

run_one() {
    local dtype="$1"
    local n_tokens="$2"
    unset MAX_LEN
    case "$dtype" in
        # fp32 KV cache needs 32 GiB for the full 131072 context (KV dtype follows
        # model dtype) which exceeds the 56G job memory -> use a smaller max len
        float32) export VLLM_CPU_KVCACHE_SPACE=10; export MAX_LEN=32768 ;;
        *)       export VLLM_CPU_KVCACHE_SPACE=20 ;;
    esac
    echo
    echo "############ DTYPE=$dtype (cpus=$SLURM_CPUS_PER_TASK, kv=${VLLM_CPU_KVCACHE_SPACE}G) ############"
    DTYPE="$dtype" N_TOKENS="$n_tokens" MODEL="$MODEL" PROMPT="$PROMPT" \
        MAX_LEN="${MAX_LEN:-}" python "$HOME/MITOS-QA/VLLM/TOOLS/bench_worker.py"
}

run_one float16 64
run_one bfloat16 64
run_one float32 64
