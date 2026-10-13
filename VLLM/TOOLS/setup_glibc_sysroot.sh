#!/bin/bash
# ---------------------------------------------------------------------------
# Rebuild the "mini sysroot" in ~/toolchain used for JIT compilation
# (torch.compile / inductor / vLLM CPU) on the ARIS compute nodes.
#
# WHY: the ARIS compute nodes (m*) run a stripped Rocky Linux 9 image with an
# EMPTY /usr/include and no glibc link-time files. Without this, g++ fails at
# vLLM startup with:
#     CppCompileError ... features.h / stdlib.h: No such file or directory
# because vLLM/torch compile CPU kernels with the system g++ at runtime.
#
# RUN THIS ON A LOGIN NODE (which has a full /usr/include).
#
# It (re)creates:
#   ~/toolchain/glibc/usr/include   <- copy of /usr/include
#   ~/toolchain/glibc/usr/lib64     <- crti.o, crtn.o, libc.so, ... link files
#   ~/toolchain/bin/gcc, ~/toolchain/bin/g++  <- wrappers adding "-idirafter"
#
# Your job script (see run_vllm_cpu.sh section 3b) then only needs:
#   export PATH=$HOME/toolchain/bin:$PATH
#   export LIBRARY_PATH=$HOME/toolchain/glibc/usr/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}
#
# Verify afterwards on a compute node:   sbatch --wait diag_compute_node.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# Toolchain used by the wrappers (needed for the verification step at the end)
module load gnu/13.3.0 2>/dev/null || true

SYSROOT="$HOME/toolchain/glibc"
BINDIR="$HOME/toolchain/bin"
PY_INC="/apps/applications/python/3.12.9/include/python3.12"

# --- Sanity check: only makes sense where the real headers exist -----------
if [ ! -f /usr/include/features.h ]; then
  echo "ERROR: /usr/include/features.h not found on this node."
  echo "Run this script on a LOGIN node (compute nodes have an empty /usr/include)."
  exit 1
fi

if [ ! -f "$PY_INC/Python.h" ]; then
  echo "WARNING: $PY_INC/Python.h not found - check the python module version!"
fi

echo "== 1. Copying /usr/include -> $SYSROOT/usr/include =="
mkdir -p "$SYSROOT/usr/include" "$SYSROOT/usr/lib64" "$BINDIR"
rsync -a --delete /usr/include/ "$SYSROOT/usr/include/"

echo "== 2. Copying link-time files from /usr/lib64 =="
for f in crti.o crtn.o crt1.o Scrt1.o rcrt1.o \
         libc.so libc_nonshared.a libm.so \
         libpthread.a libdl.a librt.a libutil.a; do
  cp -a "/usr/lib64/$f" "$SYSROOT/usr/lib64/"
done

# libc.so is a linker script pointing to /usr/lib64/libc_nonshared.a,
# which does NOT exist on compute nodes -> rewrite the path to our copy.
sed -i "s|/usr/lib64/libc_nonshared.a|$SYSROOT/usr/lib64/libc_nonshared.a|" \
    "$SYSROOT/usr/lib64/libc.so"

# These are plain runtime files on compute nodes; only the dev symlink is needed.
ln -sf /usr/lib64/libmvec.so.1   "$SYSROOT/usr/lib64/libmvec.so"
ln -sf /usr/lib64/libresolv.so.2 "$SYSROOT/usr/lib64/libresolv.so"

echo "== 3. Creating compiler wrappers in $BINDIR =="
cat > "$BINDIR/g++" <<'EOF'
#!/bin/bash
# ARIS wrapper: compute nodes have an empty /usr/include (no glibc headers),
# so JIT compilation must find them via -idirafter. -idirafter searches AFTER
# the toolchain's own dirs (exactly where /usr/include normally sits), so that
# #include_next chains (e.g. <cmath> -> <math.h>) resolve correctly.
# Do NOT use CPLUS_INCLUDE_PATH for this: it is searched too early and breaks
# include_next!
GLIBC_INC="$HOME/toolchain/glibc/usr/include"
PY_INC="/apps/applications/python/3.12.9/include/python3.12"
exec "${COMPILER_GNUROOT:-/apps/compilers/gnu/13.3.0}/bin/g++" \
     -idirafter "$GLIBC_INC" -idirafter "$PY_INC" "$@"
EOF

cat > "$BINDIR/gcc" <<'EOF'
#!/bin/bash
# ARIS wrapper: see the g++ wrapper for details.
GLIBC_INC="$HOME/toolchain/glibc/usr/include"
PY_INC="/apps/applications/python/3.12.9/include/python3.12"
exec "${COMPILER_GNUROOT:-/apps/compilers/gnu/13.3.0}/bin/gcc" \
     -idirafter "$GLIBC_INC" -idirafter "$PY_INC" "$@"
EOF

chmod +x "$BINDIR/g++" "$BINDIR/gcc"

echo "== 4. Verification =="
ls -l "$SYSROOT/usr/include/features.h"
ls -l "$BINDIR/"
printf '#include <features.h>\nint main(){return 0;}\n' > /tmp/sysroot_test.cpp
if "$BINDIR/g++" -E /tmp/sysroot_test.cpp >/dev/null 2>&1; then
  echo "features.h preprocess via wrapper: OK"
else
  echo "features.h preprocess via wrapper: FAIL"
fi
printf '#include <vector>\n#include <cmath>\nint main(){return 0;}\n' > /tmp/sysroot_test.cpp
if "$BINDIR/g++" -fPIC -shared /tmp/sysroot_test.cpp -o /tmp/sysroot_test.so -fopenmp 2>/dev/null; then
  echo "shared library build via wrapper: OK"
else
  echo "shared library build via wrapper: FAIL"
fi
rm -f /tmp/sysroot_test.cpp /tmp/sysroot_test.so

echo
echo "Done. In your job script add:"
echo '  export PATH=$HOME/toolchain/bin:$PATH'
echo '  export LIBRARY_PATH=$HOME/toolchain/glibc/usr/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}'
echo "Then verify on a compute node:  sbatch --wait diag_compute_node.sh"
