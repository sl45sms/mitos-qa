# Create a Question Answering Pairs Dataset
This repository generates a question-answering pairs dataset based on the MITOS dataset and the Greek Code of Municipalities and Communities.

The dataset is intended for later use on argilla.glossapi.gr.
The script is designed to run on the ARIS Greek HPC on THIN (CPU) nodes, generating the QA pairs with the Greek LLM Llama-Krikri-8B-Instruct (ILSP) served by vLLM.
Single-stream CPU decoding is slow (a few tokens per second), but bulk generation (fp32, 128–256 parallel requests) sustains ~35–49 tokens/s on a single-rank server and up to ~128 tokens/s on a full node — see [ARCHITECTURE.md](ARCHITECTURE.md) §6.

Processing around 5000 documents on CPU nodes will therefore take considerable time. (~18 hours)
Please be patient while the processing is ongoing.

Look at [ARCHITECTURE.md](ARCHITECTURE.md) for details on the system architecture and generation flow.

## Preparation
Always use the CPU environment for running Llama-Krikri on ARIS Greek HPC.
```bash
cd $HOME/MITOS-QA/VLLM
source ./vllm_cpu_env/bin/activate
```
## Run  
### Interactive - fastest single-stream:
`VLLM_TP=8 sbatch --cpus-per-task=128 --mem=160G --time=4:00:00 run_vllm_cpu.sh`

### Bulk QA - max throughput: (128 to 256 parallel requests)
`VLLM_TP=8 VLLM_DTYPE=float32 VLLM_MAX_LEN=16384 VLLM_CPU_KVCACHE_SPACE=8 \
    sbatch --cpus-per-task=128 --mem=160G --time=24:00:00 run_vllm_cpu.sh`

### Bulk QA - resource-conscious (only ~20-32 cores)
`VLLM_DTYPE=float32 VLLM_MAX_LEN=32768 VLLM_CPU_KVCACHE_SPACE=16 sbatch run_vllm_cpu.sh`

### Generate QA pairs
first enable the python environment:
```
cd $HOME/MITOS-QA/VLLM
source ./vllm_cpu_env/bin/activate
```
start the bulk QA script above to initialize vllm
and then run the generate_qa.py script:
```bash
python generate_qa.py \
    --input data/merged_unfiltered_dataset.jsonl \
    --output data/mitos_qa.jsonl \
    --concurrency 64 --samples 5 --max-tokens 1024
```

### Tools
- setup_glibc_sysroot.sh	Recreate ~/toolchain sysroot
- diag_compute_node.sh	diagnostic for compute nodes ("torch.compile OK")
- smoke_test_serve.sh	End-to-end smoke test of the server
- forward_port.sh	Forward a port from the compute node to the login node
