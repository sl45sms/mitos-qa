#!/bin/bash
#SBATCH --job-name=vllm_smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --account=pr020006_thin
#SBATCH --output=vllm_smoke_%j.out
#SBATCH --error=vllm_smoke_%j.err

# Smoke test: start the vLLM server, wait for /health, send one chat request, exit.
export WORKDIR=/work/pr020/pskarvel
module load gnu/13.3.0
module load python/3.12.9
source ~/MITOS-QA/VLLM/vllm_cpu_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VLLM_CPU_KVCACHE_SPACE=10
export VLLM_CPU_OMP_THREADS_BIND=auto
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# --- ARIS fix: compute nodes have an empty /usr/include (no glibc headers),
# --- so JIT compilation must use the wrapper compilers + sysroot copy.
export PATH=$HOME/toolchain/bin:$PATH
export LIBRARY_PATH=$HOME/toolchain/glibc/usr/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}

# Fresh compile cache (a previous failed run may have left partial entries)
rm -rf $HOME/.cache/vllm/torch_compile_cache

vllm serve ${WORKDIR}/models/Mistral-7B-Instruct-v0.3 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 &
PID=$!

UP=0
for i in $(seq 1 150); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "SERVER UP after $((i*10)) seconds"
    UP=1
    break
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "SERVER DIED - see errors above"
    wait $PID
    exit 1
  fi
  sleep 10
done

if [ "$UP" = "1" ]; then
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"/work/pr020/pskarvel/models/Mistral-7B-Instruct-v0.3","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}'
  echo
  echo "SMOKE TEST: OK"
else
  echo "SMOKE TEST: server did not become ready in time"
fi

kill $PID 2>/dev/null
wait 2>/dev/null
echo "SMOKE TEST DONE"
