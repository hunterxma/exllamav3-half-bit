#!/usr/bin/env bash
# One-command install of exllamav3 with the half-rate (4.5 .. 7.5 bpw) kernels, Linux + NVIDIA:
#
#   bash install.sh                 # from a checkout of this repo
#
# Needs only the NVIDIA driver, a C++ compiler and curl or git; everything else is fetched into the repo directory.
# What it does, failing early with a plain message instead of a compiler error:
#   1. finds every GPU's compute capability and the newest CUDA the driver runs (nvidia-smi)
#   2. checks disk space and the C++ compiler; gets uv (a self-contained Python + package manager) if it is missing
#   3. picks the CUDA family: a local toolkit (nvcc) the driver runs if there is one, else the newest the driver runs;
#      creates a venv (default ./venv, or $EXL3_VENV) with the matching PyTorch build
#   4. no usable local toolkit: installs NVIDIA's pip CUDA compiler of exactly PyTorch's CUDA version into the venv
#      (no root, no download page)
#   5. compiles a tiny kernel first (catches a host compiler nvcc does not accept and tries an older g++), then the
#      extension for the GPUs in this machine only (minutes, not the hour an all-architecture build takes)
#   6. runs the self-test (python -m exllamav3.util.selftest)
#
# Overrides: EXL3_VENV=path  CUDA_HOME=/usr/local/cuda-X.Y  TORCH_SPEC="torch==2.11.0"  MAX_JOBS=N
#            EXL3_ARCH="8.6;12.0" (architectures to build)  EXL3_PIP_CUDA=1 (use the pip toolkit even if one is installed)
#            EXL3_CUDA=12.8 (PyTorch CUDA family: 12.6, 12.8, 12.9 or 13.0; implies the pip toolkit)
#            EXL3_NO_SELFTEST=1 (build only, e.g. while the GPU is busy)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
REPO=$PWD
T0=$(date +%s)
say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
ver() { printf '%d%03d' "${1%%.*}" "$(echo "${1#*.}" | cut -d. -f1)"; }   # "12.8" -> 12008, comparable as integers

[ "$(uname -s)" = Linux ] || die "this installer is for Linux (Windows: see README, build with VS Build Tools)"
[ "$(uname -m)" = x86_64 ] || [ "$(uname -m)" = aarch64 ] || die "unsupported CPU architecture $(uname -m) (x86_64 or aarch64)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver first (AMD GPUs: build with ROCm, see README)"

# 1. GPUs + driver: build for every distinct compute capability in the machine that exllamav3 supports
DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
if [ -z "$DRV_CUDA" ]; then   # "CUDA Version: N/A" (e.g. a container without libcuda): from the driver version
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  [ -n "$DRV" ] || die "could not read the driver version from nvidia-smi (is the driver loaded?)"
  DRV_CUDA=$( [ "$DRV" -ge 580 ] && echo 13.0 || { [ "$DRV" -ge 575 ] && echo 12.9; } || { [ "$DRV" -ge 570 ] && echo 12.8; } \
             || { [ "$DRV" -ge 560 ] && echo 12.6; } || echo 12.4)
  warn "nvidia-smi shows no CUDA version; driver $DRV supports CUDA $DRV_CUDA"
fi
ARCHS=""; NEED=11.4
while IFS=, read -r cap name; do
  cap=$(echo "$cap" | tr -d ' '); name=$(echo "$name" | sed 's/^ *//')
  [ -n "${EXL3_ARCH:-}" ] && break
  if [ "$(ver "$cap")" -lt "$(ver 7.5)" ]; then warn "skipping $name (compute $cap): exllamav3 needs compute capability 7.5 or newer"; continue; fi
  case ";$ARCHS;" in *";$cap;"*) ;; *) ARCHS=${ARCHS:+$ARCHS;}$cap ;; esac
  say "GPU: $name (compute $cap)"
done < <(nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader)
ARCHS=${EXL3_ARCH:-$ARCHS}
[ -n "$ARCHS" ] || die "no supported NVIDIA GPU found (exllamav3 needs compute capability 7.5 or newer: RTX 20xx / T4 and later)"
for cap in ${ARCHS//;/ }; do
  case "$cap" in
    12.*|10.*|11.*) n=12.8 ;;   # Blackwell
    9.*) n=12.0 ;;              # Hopper
    8.9) n=11.8 ;;              # Ada
    *) n=11.4 ;;                # Ampere, Turing
  esac
  [ "$(ver $n)" -gt "$(ver $NEED)" ] && NEED=$n
done
say "driver supports CUDA $DRV_CUDA; building for compute $ARCHS (needs CUDA $NEED or newer)"

# 2. Preflight: disk, C++ compiler, uv
FREE_GB=$(df -Pk "$REPO" | awk 'NR == 2 {print int($4 / 1048576)}')
[ "$FREE_GB" -ge 8 ] || die "only $FREE_GB GB free in $REPO; the install needs about 12 GB (PyTorch, CUDA compiler, build)"
[ "$FREE_GB" -ge 15 ] || warn "$FREE_GB GB free in $REPO; the install needs about 12 GB"
command -v g++ >/dev/null || command -v c++ >/dev/null || die "no C++ compiler found (Debian/Ubuntu: sudo apt install build-essential; Fedora: sudo dnf install gcc-c++)"
UV=$(command -v uv || true)
[ -z "$UV" ] && [ -x "$REPO/.uv/uv" ] && UV=$REPO/.uv/uv
if [ -z "$UV" ]; then
  # uv brings its own Python (with headers and venv), so no system Python packages are needed
  say "getting uv into $REPO/.uv (Python + package manager, ~40 MB, nothing installed system-wide)"
  if command -v curl >/dev/null; then GET=(curl -LsSf); elif command -v wget >/dev/null; then GET=(wget -qO-); else GET=(); fi
  [ ${#GET[@]} -gt 0 ] && "${GET[@]}" https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$REPO/.uv" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null 2>&1 \
    && UV=$REPO/.uv/uv
  [ -x "${UV:-}" ] || warn "could not get uv; falling back to the system Python"
fi

# 3. CUDA family + venv + PyTorch. A local toolkit the driver runs keeps its family (PyTorch must match nvcc's major);
#    otherwise the newest family the driver runs, and the pip toolkit below.
idx_of() { case "$1" in 13.*) echo cu130 ;; 12.9) echo cu129 ;; 12.8) echo cu128 ;; 12.6|12.7) echo cu126 ;; *) echo "" ;; esac; }
LOCAL=""; LOCALV=""; best=0
cands=()
[ -n "${CUDA_HOME:-}" ] && cands+=("$CUDA_HOME/bin/nvcc")
for d in /usr/local/cuda-*/bin/nvcc /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do [ -x "$d" ] && cands+=("$d"); done
command -v nvcc >/dev/null && cands+=("$(command -v nvcc)")
if [ -z "${EXL3_PIP_CUDA:-}" ] && [ -z "${EXL3_CUDA:-}" ]; then
  for n in "${cands[@]}"; do
    v=$("$n" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    [ -n "$v" ] && [ -n "$(idx_of "$v")" ] || continue
    if [ -n "${CUDA_HOME:-}" ] && [ "$n" = "$CUDA_HOME/bin/nvcc" ]; then
      [ "$(ver "$v")" -ge "$(ver $NEED)" ] || die "CUDA_HOME=$CUDA_HOME is CUDA $v; compute $ARCHS needs $NEED or newer"
      LOCAL=$n; LOCALV=$v; break
    fi
    if [ "$(ver "$v")" -ge "$(ver $NEED)" ] && [ "$(ver "$v")" -le "$(ver "$DRV_CUDA")" ] && [ "$(ver "$v")" -gt "$best" ]; then
      LOCAL=$n; LOCALV=$v; best=$(ver "$v")
    fi
  done
fi
if [ -n "$LOCAL" ]; then
  FAMILY=$LOCALV
  say "CUDA toolkit: $(dirname "$(dirname "$LOCAL")") (nvcc $LOCALV)"
else
  FAMILY=""
  for f in ${EXL3_CUDA:-13.0 12.9 12.8 12.6}; do
    [ -n "$(idx_of "$f")" ] || die "EXL3_CUDA=$f: PyTorch builds exist for CUDA 12.6, 12.8, 12.9 and 13.0"
    if [ "$(ver $f)" -le "$(ver "$DRV_CUDA")" ] && [ "$(ver $f)" -ge "$(ver $NEED)" ]; then FAMILY=$f; break; fi
  done
  [ -n "$FAMILY" ] || [ -z "${EXL3_CUDA:-}" ] || die "EXL3_CUDA=$EXL3_CUDA: needs CUDA $NEED..$DRV_CUDA (GPU minimum..driver)"
  [ -n "$FAMILY" ] || die "the driver supports CUDA $DRV_CUDA; this build needs a driver for CUDA $( [ "$(ver $NEED)" -gt "$(ver 12.6)" ] && echo "$NEED" || echo 12.6 ) or newer: update the NVIDIA driver"
  [ -n "${EXL3_CUDA:-}${EXL3_PIP_CUDA:-}" ] && say "CUDA family $FAMILY; NVIDIA's pip CUDA compiler will be installed in the venv" \
    || say "no local CUDA toolkit this driver runs (found: ${cands[*]:-none}); NVIDIA's pip CUDA compiler will be installed in the venv"
fi
IDX=$(idx_of "$FAMILY")
TORCH_SPEC=${TORCH_SPEC:-torch}
VENV=${EXL3_VENV:-$REPO/venv}
if [ -n "${UV:-}" ]; then
  # cache and venv on different filesystems: copy (uv would warn on every install that hardlinking failed)
  UVC=$("$UV" cache dir 2>/dev/null || true); mkdir -p "$UVC" "$(dirname "$VENV")" 2>/dev/null || true
  [ -n "$UVC" ] && [ "$(stat -c %d "$UVC" 2>/dev/null)" != "$(stat -c %d "$(dirname "$VENV")")" ] && export UV_LINK_MODE=${UV_LINK_MODE:-copy}
  [ -x "$VENV/bin/python" ] || "$UV" venv -q -p 3.12 "$VENV" || die "uv could not create the venv at $VENV"
  PIP=("$UV" pip install -p "$VENV/bin/python")
else
  PY=$(command -v python3.12 || command -v python3.13 || command -v python3.11 || command -v python3.14 || command -v python3.10 || command -v python3 || true)
  [ -n "$PY" ] || die "no python3 found (3.10 .. 3.14 supported), and uv could not be fetched"
  "$PY" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 14)' 2>/dev/null || die "$PY is $("$PY" -V 2>&1); 3.10 .. 3.14 is supported"
  "$PY" -c 'import sysconfig, os, sys; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' \
    || die "$PY has no development headers (Debian/Ubuntu: sudo apt install python3-dev python3-venv), or install uv: https://docs.astral.sh/uv/"
  [ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV" || die "python venv module missing (Debian/Ubuntu: sudo apt install python3-venv)"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  PIP=("$VENV/bin/python" -m pip install)
fi
say "venv: $VENV; installing $TORCH_SPEC ($IDX)"
"${PIP[@]}" "$TORCH_SPEC" numpy --index-url "https://download.pytorch.org/whl/$IDX" >/dev/null   # numpy now: torch warns on import without it
TCUDA=$("$VENV/bin/python" -c 'import torch; print(torch.version.cuda or "")')
[ -n "$TCUDA" ] || die "the installed PyTorch has no CUDA support (index $IDX)"

# 4. CUDA toolkit: the local one, or NVIDIA's own packages in the venv (no root). Version: the newest release of
#    PyTorch's CUDA major the driver runs (only the major must match PyTorch; an older minor can clash with a newer
#    glibc's math headers). CUDA 13+: the pip cuda-toolkit metapackage (one coherent release) in its own directory,
#    never over PyTorch's CUDA packages. CUDA 12: NVIDIA's redist compiler archives + PyTorch's CUDA wheels.
pip_toolkit() {   # CUDA 13+: NVIDIA's pip cuda-toolkit. CUDA 12: its pip wheel has no nvcc, so NVIDIA's redist archives
  local CV TK c s
  CV=$("$VENV/bin/python" - "$TCUDA" "$DRV_CUDA" <<'PY'
import json, re, sys, urllib.request
torch_cuda, drv = sys.argv[1], sys.argv[2]
key = lambda v: tuple(int(x) for x in v.split(".")[:2])
fit = lambda v: v.split(".")[0] == torch_cuda.split(".")[0] and key(torch_cuda) <= key(v) <= key(drv)
try:
    if int(torch_cuda.split(".")[0]) >= 13:
        rel = json.load(urllib.request.urlopen("https://pypi.org/pypi/cuda-toolkit/json", timeout=30))["releases"]
        ok = [v for v, f in rel.items() if f and re.fullmatch(r"\d+\.\d+\.\d+(\.\d+)?", v) and fit(v)]
        ok = sorted({".".join(v.split(".")[:2]) for v in ok}, key=key)
    else:   # full x.y.z of the newest redist release
        page = urllib.request.urlopen("https://developer.download.nvidia.com/compute/cuda/redist/", timeout=30).read().decode()
        ok = sorted({v for v in re.findall(r"redistrib_(\d+\.\d+\.\d+)\.json", page) if fit(v)},
                    key=lambda v: tuple(int(x) for x in v.split(".")))
    print(ok[-1] if ok else torch_cuda)
except Exception:
    print(torch_cuda)
PY
)
  TK=$VENV/cuda-toolkit-$CV
  say "installing NVIDIA's CUDA $CV compiler into $TK (PyTorch is CUDA $TCUDA)"
  CUDA_HOME=$VENV/cuda; rm -rf "$CUDA_HOME"; mkdir -p "$CUDA_HOME/bin" "$CUDA_HOME/lib"
  if [ "${CV%%.*}" -ge 13 ]; then
    [ -x "$TK/nvidia/cu${CV%%.*}/bin/nvcc" ] || \
      "${PIP[@]}" -q --target "$TK" "cuda-toolkit[nvcc,cccl,cudart,curand,cublas,cusparse,cusolver]==$CV.*" \
      || die "could not install the CUDA $CV compiler from pip"
    for c in "$TK"/nvidia/*/; do
      [ -d "${c}bin" ] && cp -rf "${c}bin/." "$CUDA_HOME/bin/"
      for s in include lib nvvm; do [ -d "$c$s" ] && cp -rsf "$c$s" "$CUDA_HOME/" 2>/dev/null; done
    done
  else
    # compiler (nvcc + NVVM + CRT) and CCCL from NVIDIA's redistributable archives, checked against the manifest's
    # sha256; runtime, cuBLAS, cuSPARSE, cuRAND and cuSOLVER headers and libraries come with PyTorch's own wheels
    [ -x "$TK/cuda_nvcc/bin/nvcc" ] || "$VENV/bin/python" - "$CV" "$TK" <<'PY' || die "could not download the CUDA $CV compiler"
import hashlib, json, sys, tarfile, urllib.request, pathlib
cv, tk = sys.argv[1], pathlib.Path(sys.argv[2])
base = "https://developer.download.nvidia.com/compute/cuda/redist/"
man = json.load(urllib.request.urlopen(f"{base}redistrib_{cv}.json", timeout=60))
arch = {"x86_64": "linux-x86_64", "aarch64": "linux-sbsa"}[__import__("platform").machine()]
tk.mkdir(parents=True, exist_ok=True)
for comp in ("cuda_nvcc", "cuda_cccl"):
    e = man[comp][arch]
    data = urllib.request.urlopen(base + e["relative_path"], timeout=600).read()
    if hashlib.sha256(data).hexdigest() != e["sha256"]:
        sys.exit(f"{comp}: sha256 mismatch")
    out = tk / comp; out.mkdir(exist_ok=True)
    tmp = tk / f"{comp}.tar.xz"; tmp.write_bytes(data)
    with tarfile.open(tmp) as t:
        top = t.getnames()[0].split("/")[0]
        for m in t.getmembers():
            m.name = m.name[len(top) + 1:] if m.name.startswith(top + "/") else ""
            if m.name:
                try: t.extract(m, out, filter="tar")
                except TypeError: t.extract(m, out)   # Python without extraction filters (< 3.10.12 / 3.11.4)
    tmp.unlink()
PY
    cp -rf "$TK/cuda_nvcc/bin/." "$CUDA_HOME/bin/"
    cp -rsf "$TK/cuda_nvcc/nvvm" "$CUDA_HOME/"
    local site; site=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    for c in "$TK/cuda_nvcc" "$TK/cuda_cccl" "$site"/nvidia/*; do
      for s in include lib; do [ -d "$c/$s" ] && cp -rsf "$c/$s" "$CUDA_HOME/" 2>/dev/null; done
    done
    # the build links -lcudart; PyTorch's wheel has only the versioned library
    [ -e "$CUDA_HOME/lib/libcudart.so" ] || ln -sf "$(ls "$CUDA_HOME"/lib/libcudart.so.* | head -1)" "$CUDA_HOME/lib/libcudart.so"
  fi
  [ -x "$CUDA_HOME/bin/nvcc" ] || die "the CUDA $CV toolkit has no nvcc (looked in $TK)"
  [ -e "$CUDA_HOME/lib64" ] || ln -s lib "$CUDA_HOME/lib64"
  export CUDA_HOME
}

# 5. Compile check (seconds) with the headers the extension uses, incl. <cmath> (a glibc newer than the CUDA release
#    breaks there): nvcc accepts the host compiler and builds for these GPUs. CUDA releases cap the gcc version; when the
#    default is too new, try the installed older ones. A local toolkit that fails falls back to the pip one.
PROBE=$(mktemp -d); trap 'rm -rf "$PROBE"' EXIT
printf '#include <cmath>\n#include <math.h>\n#include <cuda_runtime.h>\n#include <cooperative_groups.h>\n#include <curand_kernel.h>\n#include <cuda/std/cstdint>\n__global__ void k(float* x) { x[threadIdx.x] = rsqrtf(x[threadIdx.x]); }\n' > "$PROBE/t.cu"
GENS=(); for cap in ${ARCHS//;/ }; do GENS+=(-gencode "arch=compute_${cap/./},code=sm_${cap/./}"); done
probe() { "$CUDA_HOME/bin/nvcc" ${1:+-ccbin "$1"} "${GENS[@]}" -c "$PROBE/t.cu" -o "$PROBE/t.o" > "$PROBE/log" 2>&1; }
compiles() {   # with the default host compiler, else an older installed g++ (sets CUDAHOSTCXX)
  probe "" && return 0
  grep -qiE "unsupported (GNU|clang) version|host compiler" "$PROBE/log" || return 1
  local gxx
  for gxx in $(ls /usr/bin/g++-[0-9]* 2>/dev/null | sort -t- -k2 -rn); do
    if probe "$gxx"; then export CUDAHOSTCXX=$gxx; say "nvcc does not accept the default g++; using $gxx"; return 0; fi
  done
  return 1
}
if [ -n "$LOCAL" ]; then
  [ "${TCUDA%%.*}" = "${LOCALV%%.*}" ] || die "PyTorch was built for CUDA $TCUDA but the toolkit is $LOCALV (majors must match)"
  export CUDA_HOME=$(dirname "$(dirname "$LOCAL")")
  if ! compiles; then
    sed -n '1,4p' "$PROBE/log" >&2
    warn "the local CUDA $LOCALV toolkit cannot compile here (above); using NVIDIA's pip toolkit instead"
    pip_toolkit
  fi
else
  pip_toolkit
fi
if ! probe "${CUDAHOSTCXX:-}" && ! compiles; then
  sed -n '1,8p' "$PROBE/log" >&2
  if grep -q "mathcalls.h" "$PROBE/log"; then
    die "CUDA $("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p') headers do not compile with this system's glibc $(ldd --version | head -1 | awk '{print $NF}') (above).
  CUDA 13 does: $( [ "$(ver "$DRV_CUDA")" -ge "$(ver 13.0)" ] && echo "rerun without EXL3_CUDA / CUDA_HOME" || echo "update the NVIDIA driver to R580 or newer (CUDA 13), then rerun")"
  fi
  die "nvcc could not compile a test kernel (above). If it names an unsupported compiler version, install an older g++
  (Debian/Ubuntu: sudo apt install g++-13) and rerun, or set CUDAHOSTCXX=/usr/bin/g++-N"
fi
say "CUDA compiler: $CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'))${CUDAHOSTCXX:+, host $CUDAHOSTCXX}"
export PATH=$CUDA_HOME/bin:$PATH
"${PIP[@]}" -q setuptools wheel ninja
"${PIP[@]}" -q -r "$REPO/requirements_examples.txt"   # examples/chat.py and friends work straight after install

# 6. Build for these GPUs only. RAM bounds parallelism: each nvcc job can take ~3 GB
CORES=$(nproc); RAM_GB=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
JOBS=${MAX_JOBS:-$(( CORES < RAM_GB / 3 ? CORES : RAM_GB / 3 ))}; [ "$JOBS" -ge 1 ] || JOBS=1
LOG=$REPO/build.log
say "building the extension for compute $ARCHS with $JOBS jobs (typically 10-30 min; log: $LOG)"
if ! TORCH_CUDA_ARCH_LIST=$ARCHS MAX_JOBS=$JOBS "${PIP[@]}" --no-build-isolation -v "$REPO" > "$LOG" 2>&1; then
  grep -m5 -E "error|Error|fatal" "$LOG" >&2 || true
  grep -q -i "killed" "$LOG" && warn "a compiler process was killed (out of memory?): rerun with MAX_JOBS=$(( JOBS / 2 > 0 ? JOBS / 2 : 1 ))"
  die "build failed; the first errors are above, full log in $LOG"
fi

# JIT-compiled kernels (tests, extensions loaded at runtime) must use the same toolkit and host compiler as the build,
# not whatever nvcc is first on PATH
sed -i '/# EXL3 install.sh/,$d' "$VENV/bin/activate"
{ printf '\n# EXL3 install.sh: CUDA toolkit the extension was built with\nexport CUDA_HOME="%s"\nexport PATH="$CUDA_HOME/bin:$PATH"\n' "$CUDA_HOME"
  [ -n "${CUDAHOSTCXX:-}" ] && printf 'export CUDAHOSTCXX="%s"\n' "$CUDAHOSTCXX"; true; } >> "$VENV/bin/activate"

# 7. Self-test
MIN=$(( ($(date +%s) - T0) / 60 ))
[ -n "${EXL3_NO_SELFTEST:-}" ] && { say "built in $MIN min; self-test skipped (EXL3_NO_SELFTEST). Run: $VENV/bin/python -m exllamav3.util.selftest"; exit 0; }
say "self-test"
"$VENV/bin/python" -m exllamav3.util.selftest || die "self-test failed (see above)"
say "done in $MIN min. Activate with: source $VENV/bin/activate
    chat with an EXL3 model:  python examples/chat.py -m /path/to/model -mode <prompt format>  (python examples/chat.py -modes lists them)"
