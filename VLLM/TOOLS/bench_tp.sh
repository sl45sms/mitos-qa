#!/bin/bash
#SBATCH --job-name=bench_tp
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128     # full node
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --account=pr020006_thin
#SBATCH --output=OUTPUTS/bench_tp_%j.out
#SBATCH --error=OUTPUTS/bench_tp_%j.err
#
# Tensor-parallel CPU test - one rank per NUMA node (decode is memory-bandwidth
# bound, TP multiplies the usable bandwidth). Default = full node (TP=8, 128 cpus).
# Usage:
#   full node: sbatch --wait TOOLS/bench_tp.sh            (TP=8, all 8 NUMA nodes)
#   half node: sbatch --wait --cpus-per-task=32 --mem=64G TOOLS/bench_tp.sh 2
# Sizing: 16 CPUs (= 1 NUMA node) per rank; VLLM_CPU_KVCACHE_SPACE is PER RANK.
# References (single rank = 1 NUMA node): bf16 b=1: 2.7, b=32: 8.7; fp32 b=32: 35.2 tok/s.
# Env overrides: BF16_BATCHES / FP32_BATCHES (comma-separated) to pick the batch lists.
set -u

TP=${1:-8}
export WORKDIR=/work/pr020/pskarvel
module load gnu/13.3.0
module load python/3.12.9
source ~/MITOS-QA/VLLM/vllm_cpu_env/bin/activate

export PATH=$HOME/toolchain/bin:$PATH
export LIBRARY_PATH=$HOME/toolchain/glibc/usr/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_PRELOAD="$HOME/toolchain/preload/libtcmalloc_minimal.so.4:$HOME/MITOS-QA/VLLM/vllm_cpu_env/lib/libiomp5.so"

export VLLM_CPU_OMP_THREADS_BIND=auto
export VLLM_CPU_NUM_OF_RESERVED_CPU=$TP      # 1 core per rank for the coordinators
# (VLLM_CPU_KVCACHE_SPACE is set per phase below - it is PER RANK!)

# Auto-binding fix (see run_vllm_cpu.sh section 2a)
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
echo "TP=$TP CPU_VISIBLE_MEMORY_NODES=$CPU_VISIBLE_MEMORY_NODES (allocation: $(taskset -pc $$ | sed 's/.*: //'))"

MODEL=${WORKDIR}/models/Llama-Krikri-8B-Instruct
PROMPT='Γράψε μια σύντομη παράγραφο για τη θάλασσα.'

cd "$HOME"

run_phase() {
    local dtype="$1" kv="$2" mlen="$3" batches="$4"
    echo
    echo "############### TP=$TP PHASE $dtype (kv=${kv}G/rank, max_len=$mlen) ###############"
    TP="$TP" DTYPE="$dtype" MODEL="$MODEL" PROMPT="$PROMPT" MAX_LEN="$mlen" \
        BATCHES="$batches" VLLM_CPU_KVCACHE_SPACE="$kv" \
        python "$HOME/MITOS-QA/VLLM/TOOLS/bench_scale.py"
}

run_phase bfloat16 6 32768 "${BF16_BATCHES:-1,8,32}"
run_phase float32 8 16384 "${FP32_BATCHES:-1,32}"
