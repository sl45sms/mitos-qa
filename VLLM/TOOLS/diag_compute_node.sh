#!/bin/bash
#SBATCH --job-name=diag_node
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --account=pr020006_thin
#SBATCH --output=diag_node_%j.out
#SBATCH --error=diag_node_%j.err

echo "== hostname =="
hostname
grep PRETTY_NAME /etc/os-release 2>/dev/null
uname -r
ldd --version | head -1

echo "== /usr/include =="
ls /usr/include 2>/dev/null | wc -l
ls -l /usr/include/features.h 2>&1

echo "== glibc sysroot copy (from login node) =="
ls -l $HOME/toolchain/glibc/usr/include/features.h 2>&1
ls -l $HOME/toolchain/bin/ 2>&1

echo "== toolchain =="
module load gnu/13.3.0 python/3.12.9
export PATH=$HOME/toolchain/bin:$PATH
export LIBRARY_PATH=$HOME/toolchain/glibc/usr/lib64
echo "g++ -> $(which g++)"
g++ --version | head -1
echo "as: $(which as)"
echo "ld: $(which ld)"
ld --version 2>/dev/null | head -1

echo "== g++ search list (tail) =="
echo | g++ -E -v -x c++ - 2>&1 | sed -n "/search starts here/,/End of search list./p" | tail -13

printf '#include <features.h>\nint main(){return 0;}\n' > /tmp/fh_test.cpp
if g++ -E /tmp/fh_test.cpp >/dev/null 2>/tmp/fh_pp.err; then
  echo "PREPROCESS features.h: OK"
else
  echo "PREPROCESS features.h: FAIL"
  head -5 /tmp/fh_pp.err
fi

printf '#include <Python.h>\nint main(){return 0;}\n' > /tmp/py_test.cpp
if g++ -E /tmp/py_test.cpp >/dev/null 2>/tmp/py.err; then
  echo "PREPROCESS Python.h: OK"
else
  echo "PREPROCESS Python.h: FAIL"
  head -5 /tmp/py.err
fi

printf '#include <vector>\n#include <cmath>\nint main(){std::vector<double> v(4,1.0); return (int)(std::sqrt(v[0])*0);}\n' > /tmp/stl_test.cpp
if g++ -fPIC -shared /tmp/stl_test.cpp -o /tmp/stl_test.so -fopenmp -lpthread -ldl -lrt -lutil 2>/tmp/stl.err; then
  echo "CXX SHARED BUILD (+pthread+dl+rt+util): OK"
else
  echo "CXX SHARED BUILD: FAIL"
  head -10 /tmp/stl.err
fi

echo "== torch.compile mini test (same machinery as vLLM) =="
source $HOME/MITOS-QA/VLLM/vllm_cpu_env/bin/activate 2>/dev/null
python - <<'PY'
import sys
import sysconfig
import time
import torch

print("torch:", torch.__version__, "| python:", sys.version.split()[0])
print("sysconfig include:", sysconfig.get_paths()["include"])
print("INCLUDEPY:", sysconfig.get_config_var("INCLUDEPY"))

def f(x):
    return torch.sin(x) + x * 2.0

t0 = time.time()
cf = torch.compile(f, fullgraph=True, dynamic=False)
y = cf(torch.randn(256, 256))
print("torch.compile OK, sum=%.4f, took %.1fs" % (float(y.sum()), time.time() - t0))
PY

echo "== cpu/numa topology =="
grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/self/status
lscpu | grep -iE "model name|thread|core\(s\)|socket|numa"
echo "== cpu flags (avx512 / amx / fp16) =="
grep -o "avx512[a-z0-9_]*" /proc/cpuinfo | sort -u
grep -o "amx[a-z0-9_]*" /proc/cpuinfo | sort -u
echo "== perf libs available on compute node =="
ls -l /usr/lib64/libtcmalloc_minimal.so.4 /usr/lib64/libiomp5.so 2>&1
echo "== zentorch installed? =="
python -c "import zentorch; print('zentorch', zentorch.__version__)" 2>&1 | head -2
