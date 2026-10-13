#!/bin/bash
# Stage the libraries vLLM wants in LD_PRELOAD on ARIS compute nodes.
#
# Run this on a LOGIN node (login05), NOT on a compute node: the compute
# nodes run a stripped image without /usr/include and without libtcmalloc,
# while login nodes have both. tcmalloc is copied into ~/toolchain/preload
# (on GPFS, visible everywhere); libiomp5 ships inside the venv via the
# intel-openmp dependency and is only verified here.
#
# Idempotent: safe to re-run any time (e.g. after a cluster OS update).
set -eu

PRELOAD_DIR="$HOME/toolchain/preload"
VENV_IOMP="$HOME/MITOS-QA/VLLM/vllm_cpu_env/lib/libiomp5.so"

echo "== staging tcmalloc into $PRELOAD_DIR =="
mkdir -p "$PRELOAD_DIR"
if ls /usr/lib64/libtcmalloc_minimal.so.4* >/dev/null 2>&1; then
    cp -a /usr/lib64/libtcmalloc_minimal.so.4.5.9 /usr/lib64/libtcmalloc_minimal.so.4 "$PRELOAD_DIR"/
    echo "copied from /usr/lib64:"
    ls -l "$PRELOAD_DIR"/libtcmalloc_minimal.so.4*
else
    echo "ERROR: /usr/lib64/libtcmalloc_minimal.so.4* not found on this host." >&2
    echo "Install gperftools on a login node or extract libtcmalloc_minimal.so.4" >&2
    echo "from the official vLLM CPU wheel (vllm/libs/libtcmalloc_minimal*.so*)." >&2
    exit 1
fi

echo
echo "== checking libiomp5 (intel-openmp -> ships with the venv) =="
if [ -f "$VENV_IOMP" ]; then
    ls -l "$VENV_IOMP"
else
    echo "ERROR: $VENV_IOMP not found." >&2
    echo "Fix with:  source ~/MITOS-QA/VLLM/vllm_cpu_env/bin/activate && pip install intel-openmp==2024.2.1" >&2
    exit 1
fi

echo
echo "== dependency check (must resolve against compute-node libs: /lib64) =="
ldd "$PRELOAD_DIR/libtcmalloc_minimal.so.4.5.9" | sed 's/^/  tcmalloc: /'
ldd "$VENV_IOMP" | sed 's/^/  libiomp5: /'

echo
echo "OK. run_vllm_cpu.sh will LD_PRELOAD:"
echo "  $PRELOAD_DIR/libtcmalloc_minimal.so.4"
echo "  $VENV_IOMP"
