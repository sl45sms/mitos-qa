The aim is to create a question answering pairs dataset based on the MITOS dataset, for later use on argilla.glossapi.gr . This script are made to run on ARIS Greek HPC.
Because we do not have many GPU resources, we will be going to be running QWEN on CPU nodes instead.

We are expecting to get around 10 tokens from Qwen 3 30B MoE per input.

So for around 5000 docs will be taking some time to process on CPU nodes.
Please be patient while the processing is ongoing.


## preparation
use always the CPU environment for running QWEN on ARIS Greek HPC.
```
cd $HOME/MITOS-QA/VLLM
source ./vllm_cpu_env/bin/activate
```

## Run  
### Interactive - fastest single-stream:
VLLM_TP=8 sbatch --cpus-per-task=128 --mem=160G --time=4:00:00 run_vllm_cpu.sh

### Bulk QA - max throughput: (128 to 256 parallel requests)
VLLM_TP=8 VLLM_DTYPE=float32 VLLM_MAX_LEN=16384 VLLM_CPU_KVCACHE_SPACE=8 \
    sbatch --cpus-per-task=128 --mem=160G --time=24:00:00run_vllm_cpu.sh

### Bulk QA - resource-conscious (only ~20-32 cores)
VLLM_DTYPE=float32 VLLM_MAX_LEN=32768 VLLM_CPU_KVCACHE_SPACE=16 sbatch run_vllm_cpu.sh




# tools
setup_glibc_sysroot.sh	Recreate ~/toolchain sysroot 
diag_compute_node.sh	diagnostic for compute nodes ("torch.compile OK")
smoke_test_serve.sh	End-to-end smoke test of the server
