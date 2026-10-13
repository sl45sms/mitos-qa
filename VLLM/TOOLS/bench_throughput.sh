#!/bin/bash
#SBATCH --job-name=bench_thr
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=56G
#SBATCH --time=00:45:00
#SBATCH --account=pr020006_thin
#SBATCH --output=OUTPUTS/bench_thr_%j.out
#SBATCH --error=OUTPUTS/bench_thr_%j.err
#
# Batching throughput test: same config as run_vllm_cpu.sh, offline batching.
# Reference: single-stream bf16 = ~2.7 tok/s (job 171392).
set -u

export WORKDIR=/work/pr020/pskarvel
module load gnu/13.3.0
module load python/3.12.9
source ~/MITOS-QA/VLLM/vllm_cpu_env/bin/activate

export PATH=$HOME/toolchain/bin:$PATH
export LIBRARY_PATH=$HOME/toolchain/glibc/usr/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_PRELOAD="$HOME/toolchain/preload/libtcmalloc_minimal.so.4:$HOME/MITOS-QA/VLLM/vllm_cpu_env/lib/libiomp5.so"

export VLLM_CPU_OMP_THREADS_BIND=auto
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
export VLLM_CPU_KVCACHE_SPACE=20

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
echo "CPU_VISIBLE_MEMORY_NODES=$CPU_VISIBLE_MEMORY_NODES (allocation: $(taskset -pc $$ | sed 's/.*: //'))"

MODEL=${WORKDIR}/models/Llama-Krikri-8B-Instruct
PROMPT='Γράψε μια σύντομη παράγραφο για τη θάλασσα.'

cd "$HOME"
MODEL="$MODEL" PROMPT="$PROMPT" N_TOKENS=64 DTYPE=bfloat16 \
    python "$HOME/MITOS-QA/VLLM/TOOLS/bench_throughput.py"
