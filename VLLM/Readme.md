# Instructions to build VLLM for use on ARIS Greek HPC.

## Load necessary compiler and Python modules available on ARIS

module load python/3.12.9 # Make sure python is between 3.10 and 3.13 use `module spider python` to check available versions

## Create a virtual environment on the shared GPFS storage
cd $HOME/MITOS-QA/VLLM
python3 -m venv ./vllm_cpu_env
source ./vllm_cpu_env/bin/activate

## Install vLLM from source optimized for CPU
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --upgrade pip setuptools wheel packaging packages numpy
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu # to help ensure CPU-only installation

## downgrade click to resolve dependency conflict of cluster
pip install "click<8.0,>=7.0"

## Install CPU-only dependencies
pip install -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu # Ensure CPU-only dependencies are installed, but still allow installation from the default PyPI if needed. In case some packages like `torch` have the same version number, then pip prefers the default PyPI version.So install `torch` explicitly from the CPU-only index first.

# Install additional build dependencies for Rust extensions
pip install setuptools-rust wheel
pip install setuptools_scm cmake ninja packaging

## Build and install vLLM with the specified environment
module unload gnu
module load gnu/13.3.0  # Or the highest available GCC 12+ version use `module avail gnu` to check available versions
module load rust/1.98
export VLLM_TARGET_DEVICE=cpu
export MAX_JOBS=4  # keep it low to avoid crashing the login node due to memory usage
pip install --no-build-isolation -e .

## download model
on work directory, download the model to the GPFS storage, for example:
```bash
mkdir -p ${WORKDIR}/models/
pip install huggingface_hub
hf download ilsp/Llama-Krikri-8B-Instruct --local-dir ${WORKDIR}/models/Llama-Krikri-8B-Instruct
```

## Prepare the "mini sysroot" for the compute nodes (needed once)

The ARIS compute nodes run a stripped OS image with an **empty `/usr/include`** (no glibc
development headers). vLLM compiles CPU kernels at startup with `torch.compile`/inductor,
so without these headers the server crashes with:

```
CppCompileError ... features.h: No such file or directory
```

Create (or recreate at any time) the "mini sysroot" in `~/toolchain` — **run this on a
login node**, not on a compute node:

```bash
bash $HOME/MITOS-QA/VLLM/setup_glibc_sysroot.sh
```

The script copies the needed headers and linker files from the login node, creates
`g++`/`gcc` wrapper compilers under `~/toolchain/bin` (they use `-idirafter` to emulate a
full `/usr/include`), and self-verifies with a small compile test. It is idempotent, so
it is safe to re-run whenever needed (e.g. after a cluster OS update).

`run_vllm_cpu.sh` already activates the toolchain in its section 3b (`PATH` to the
wrappers + `LIBRARY_PATH` to the copied linker files) — no manual action needed there.

Verify on a compute node:

```bash
sbatch --wait diag_compute_node.sh   # expect "torch.compile OK"
```

Troubleshooting fallback: adding `--enforce-eager` to `vllm serve` skips JIT compilation
entirely (the server then starts without needing a C++ compiler, but CPU inference is
slower).

## Performance tuning (compute nodes: AMD EPYC 7763 "Milan")

!!! info "Hardware specs: use doc25, not the old site"
    The current cluster docs are at
    [doc25.aris.grnet.gr/system/hardware](https://doc25.aris.grnet.gr/system/hardware/):
    48 thin nodes m01-m48 (Dell R6525, 2x AMD EPYC 7763, 128 cores, 512 GB RAM),
    fat f01-f16 (1024 GB), taskp f11-f16, gpu a01-a03 (4x A100 80GB), login nodes
    2x EPYC 7413. The OLD site (doc.aris.grnet.gr) documents the retired 2015 machine
    (thin: 2x Xeon E5-2680v2, 20 cores, 64 GB) — do not use it for this cluster.
    SLURM sanity check: `sinfo -p compute -o "%n %c %m"` → 128 CPUs, 507904 MB
    (512 GB nominal, ~496 GiB visible).

The compute nodes are dual **EPYC 7763** (Zen 3, **AVX2 only — no AVX-512, no native
BF16**), 128 cores total, 8 NUMA nodes of 16 cores each (~63 GiB RAM per NUMA node,
~496 GiB per node). Decode speed is memory-bandwidth bound, so these settings matter a
lot (all already applied in `run_vllm_cpu.sh`):

1. **Stage the LD_PRELOAD libraries once (login node):**

   ```bash
   bash $HOME/MITOS-QA/VLLM/TOOLS/setup_preload_libs.sh
   ```

   It copies `libtcmalloc_minimal.so.4` from the login node into `~/toolchain/preload/`
   (compute nodes don't have it) and verifies `libiomp5.so` inside the venv. The server
   logs a clear warning if any of these is missing from `LD_PRELOAD`.

2. **Pick the dtype by workload** (measured 2026-10-13, idle node m35, one NUMA node,
   15 threads; `TOOLS/bench_scale.sh`):

   | concurrency | `bfloat16` | `float32` |
   |---|---|---|
   | 1 | **2.7 tok/s** | 1.4 tok/s |
   | 8 | 8.7 | 10.7 |
   | 32 | 8.7 (plateau) | **35.2** |
   | 64 | — | **47.4** |
   | 128 | — | **49.4** (saturates) |

   Why: on these AVX2-only Zen 3 nodes, oneDNN can only build **fp32** matmul
   primitives (`CPU unquantized GEMM dispatch: using oneDNN onednn_mm`). fp16/bf16
   always fall back to torch (`... using torch.nn.functional.linear (fallback)`) and
   plateau at ~8.7 tok/s aggregate no matter the batch size.

   - **Mode A — interactive chat:** `--dtype bfloat16` (fastest single request,
     ~2.7 tok/s; reads 16 GB of weights per token vs 32 GB for fp32).
   - **Mode B — bulk QA / evaluation:** `--dtype float32` + **client-side concurrency
     32-128** (continuous batching) → **35-49 tok/s aggregate**. Requires
     `--max-model-len 32768` (fp32 KV = 32 GiB at 131072), `VLLM_CPU_KVCACHE_SPACE≈16`
     and `--mem≥64G`.</br>
     Launch example: `VLLM_DTYPE=float32 VLLM_MAX_LEN=32768 VLLM_CPU_KVCACHE_SPACE=16 sbatch run_vllm_cpu.sh`

   fp32 adds no quality (the model is bf16-native — fp32 is a pure upcast).

3. **Do NOT export `OMP_NUM_THREADS`** (nor `MKL_/OPENBLAS_` thread vars): vLLM manages
   the OpenMP environment itself (pins OMP threads to one NUMA node, reserves 1 core,
   drops torch intra-op threads to 1). An external `OMP_NUM_THREADS` disables the last
   optimization and oversubscribes the CPU quota.

   **Gotcha (already fixed in the scripts):** vLLM's `auto` binding selects the FIRST
   allowed NUMA node, but ARIS SLURM keeps `Mems_allowed_list=0-7` even when the allocated
   CPUs are elsewhere, so vLLM picked NUMA node 0, found no allocated cores there and
   logged `core ids=[]` ("Selected CPU core number (0) should be greater than reserved CPU
   core number (1)") — i.e. **no thread pinning at all**. `run_vllm_cpu.sh` and
   `bench_dtype.sh` now export `CPU_VISIBLE_MEMORY_NODES` computed from the actual CPU
   allocation. If you launch vLLM manually on a partially-allocated node, copy that export.

4. **Verify in the startup log:**
   - `OpenMP thread binding info ... core ids=[...]` → binding is active (must NOT be empty)
   - `Reducing Torch threads ... to 1 for serving`
   - no `libtcmalloc is not found in LD_PRELOAD` (the `Failed to create oneDNN linear`
     warning is EXPECTED on these AVX2 nodes — the torch fallback is normal here)

5. **Benchmark after changes:**

   ```bash
   sbatch --wait TOOLS/bench_dtype.sh     # float16 vs bfloat16 vs float32
   ```

6. **Send requests in parallel** (continuous batching). This is where the big win is:
   with fp32, aggregate throughput grows 1.4 → 10.7 → 35 → 49 tok/s as client
   concurrency goes 1 → 8 → 32 → 128 (numbers above). For a QA dataset: either pass a
   list of prompts to the offline API (`LLM.generate([...])`), or fire concurrent HTTP
   requests — e.g. verify a running server with
   `vllm bench serve --backend openai-chat --base-url http://<node>:8000 --max-concurrency 64 ...`

### Tensor parallelism across NUMA nodes — VALIDATED (full node)

A single rank can only use ONE NUMA node. With `--tensor-parallel-size 8` (one rank per
NUMA node, full node) the measured results (job 171488, idle node; the per-rank binding is
confirmed in the startup log — each rank gets its own 15 cores):

| aggregate tok/s | b=1 | b=8 | b=32 | b=64 | b=128 | b=256 |
|---|---|---|---|---|---|---|
| single-rank bf16 | 2.7 | 8.7 | 8.7 | — | — | — |
| **TP=8 bf16** | **6.1** | **31.1** | **46.1** | 47.7 | 49.5 | — |
| single-rank fp32 | 1.4 | 10.7 | 35.2 | 47.4 | 49.4 | — |
| **TP=8 fp32** | 4.6 | — | 81.9 | 106.8 | 114.5 | **128.4** |

Ceilings: **TP=8 bf16 saturates ≈50 tok/s** (b≈64; torch-fallback bound). **TP=8 fp32 is
still climbing at b=256 (128.4 tok/s, +5% per batch doubling; busy cores drop 108→75 —
becoming all-reduce-bound; practical ceiling ≈130-150)**. For a resource-conscious bulk
run, single-rank fp32 + concurrency 128 gives 49 tok/s with only ~20-32 cores instead of
the whole node.

Usage (env overrides pass through SLURM):

```bash
# interactive (fastest single stream):
VLLM_TP=8 sbatch --cpus-per-task=128 --mem=160G run_vllm_cpu.sh
# bulk QA (float32; KV is PER RANK with TP>1, so mind max_len/KV/mem):
VLLM_TP=8 VLLM_DTYPE=float32 VLLM_MAX_LEN=16384 VLLM_CPU_KVCACHE_SPACE=8 \
    sbatch --cpus-per-task=128 --mem=160G run_vllm_cpu.sh
```

Notes:
- `VLLM_CPU_KVCACHE_SPACE` is **PER RANK** with TP>1: the script now auto-defaults it
  to 4 GiB/rank for TP>1 (the single-rank default of 20 GiB × 8 ranks = 160 GiB of KV
  caused a cgroup OOM kill in the first full-node server run, job 171567).
- `--mem=160G` covers the transient weight-load peak (each of the 8 ranks reads the
  full 15.5 GiB checkpoint concurrently ≈ 125 GiB). Raise it if you ever see OOM.
- no explicit thread-bind string is needed — the `CPU_VISIBLE_MEMORY_NODES` fix
  handles the per-rank binding automatically.
- init takes longer (8 ranks compile: 807 s bf16 / 378 s fp32 in the test); afterwards
  the AOT cache makes restarts much faster.
- the `Unknown vLLM environment variable detected: VLLM_TP/VLLM_DTYPE/VLLM_MAX_LEN`
  warnings are harmless (the script now unsets them before launching vLLM).
- `TOOLS/bench_tp.sh` reproduces the whole test.

## start the vLLM server
sbatch run_vllm_cpu.sh


## Check the server status and logs
Get status of the submitted job using:
```bash
squeue -u $USER
```
Once the job changes from PD (Pending) to R (Running) status, 
check with `squeue -u $USER` to verify what local node string (e.g., mXX) it is bound to. You can query the server internally from a different login or task node on the same ARIS network:
## Query the running vLLM server
```bash
curl http://m12:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${WORKDIR}"'/models/Llama-Krikri-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}]
  }'
```

## Access the server from your PC (VS Code port forwarding)

VS Code's **Ports** panel can only forward ports that listen on the machine it is
connected to (login05) — not on a compute node (m42). Bridge the gap with a small
relay that runs on the login node:

```bash
# on the login node (e.g. in the VS Code terminal)
bash TOOLS/forward_port.sh            # port 8000; node auto-detected from the RUNNING vllm_cpu job
bash TOOLS/forward_port.sh 8000 m42   # ...or pass port and node explicitly
```

Then in VS Code: `Ctrl/Cmd+Shift+P` → **Ports: Focus on Ports View** → **Forward a Port**
→ `8000` (VS Code usually auto-detects it as well) and open `http://localhost:8000`
on your PC.

How it works: `TOOLS/port_relay.py` listens on `login05:127.0.0.1:8000` and forwards
every connection to `<node>:8000` (compute nodes are not reachable from your PC, and
ssh to them is blocked, but login05 can reach them). VS Code then forwards that
login05 port to your PC. The relay is detached (`nohup`; log `~/.forward_port_8000.log`,
stop it with `kill $(cat ~/.forward_port_8000.pid)`), so it survives VS Code
reconnects — but **re-run `TOOLS/forward_port.sh` after every new job**, the job may
land on a different node. It can be started before the server is ready; connections
simply fail until vLLM is up.

Alternative without a relay (for plain `ssh` use; login05 resolves the tunnel
endpoint, no sshd is needed on the compute node) — add to `~/.ssh/config` on your PC:

```
Host <your-aris-login-alias>
    LocalForward 8000 m42:8000
```
