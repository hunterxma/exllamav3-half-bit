#!/usr/bin/env bash
# One-command install of exllamav3 with the half-rate (4.5 .. 7.5 bpw) kernels, Linux + NVIDIA:
#
#   bash install.sh                 # from a checkout of this repo
#
# What it does, failing early with a plain message instead of a compiler error:
#   1. finds the GPU and its compute capability (nvidia-smi) and the newest CUDA the driver supports
#   2. finds a CUDA toolkit (nvcc) new enough for that GPU and not newer than the driver
#   3. creates a venv (default ./venv, or $EXL3_VENV) and installs the matching PyTorch build
#   4. compiles the extension for this GPU only (minutes, not the hour an all-architecture build takes)
#   5. runs the self-test (python -m exllamav3.util.selftest)
#
# Overrides: EXL3_VENV=path  CUDA_HOME=/usr/local/cuda-X.Y  TORCH_SPEC="torch==2.11.0"  MAX_JOBS=N  EXL3_ARCH=12.0
#            EXL3_NO_SELFTEST=1 (build only, e.g. while the GPU is busy)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
REPO=$PWD
say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
ver() { printf '%d%03d' "${1%%.*}" "${1#*.}"; }   # "12.8" -> 12008, comparable as integers

[ "$(uname -s)" = Linux ] || die "this installer is for Linux (Windows: see README, build with VS Build Tools)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver first"

# 1. GPU + driver
ARCH=${EXL3_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
[ -n "$ARCH" ] && [ -n "$DRV_CUDA" ] || die "could not read the GPU compute capability / driver CUDA version from nvidia-smi"
case "$ARCH" in
  12.*|10.*) NEED=12.8 ;;     # Blackwell
  9.0)       NEED=12.0 ;;     # Hopper
  8.9)       NEED=11.8 ;;     # Ada
  8.*)       NEED=11.4 ;;     # Ampere
  7.5)       NEED=11.4 ;;     # Turing
  *) die "GPU $GPU (sm_${ARCH/./}) is not supported (exllamav3 needs compute capability 7.5 or newer)" ;;
esac
say "GPU: $GPU (compute $ARCH), driver supports CUDA $DRV_CUDA"

# 2. CUDA toolkit: CUDA_HOME if given, else the newest nvcc that is >= NEED and <= the driver's CUDA
pick=""; best=0
cands=()
[ -n "${CUDA_HOME:-}" ] && cands+=("$CUDA_HOME/bin/nvcc")
for d in /usr/local/cuda-*/bin/nvcc /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do [ -x "$d" ] && cands+=("$d"); done
command -v nvcc >/dev/null && cands+=("$(command -v nvcc)")
for n in "${cands[@]}"; do
  v=$("$n" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
  [ -n "$v" ] || continue
  if [ -n "${CUDA_HOME:-}" ] && [ "$n" = "$CUDA_HOME/bin/nvcc" ]; then pick=$n; PICKV=$v; break; fi
  if [ "$(ver "$v")" -ge "$(ver $NEED)" ] && [ "$(ver "$v")" -le "$(ver "$DRV_CUDA")" ] && [ "$(ver "$v")" -gt "$best" ]; then
    pick=$n; PICKV=$v; best=$(ver "$v")
  fi
done
if [ -z "$pick" ]; then
  found=$(for n in "${cands[@]}"; do printf '%s ' "$n ($("$n" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p'))"; done)
  die "no CUDA toolkit between $NEED (needed for compute $ARCH) and $DRV_CUDA (driver) was found.
  Found: ${found:-none}
  Install one from https://developer.nvidia.com/cuda-downloads (toolkit only), or set CUDA_HOME=/path/to/cuda"
fi
[ "$(ver "$PICKV")" -ge "$(ver $NEED)" ] || die "CUDA_HOME=$CUDA_HOME is CUDA $PICKV; compute $ARCH needs $NEED or newer"
export CUDA_HOME=$(dirname "$(dirname "$pick")"); export PATH=$CUDA_HOME/bin:$PATH
say "CUDA toolkit: $CUDA_HOME (nvcc $PICKV)"

# 3. venv + PyTorch built for the same CUDA major.minor family as the toolkit
case "$PICKV" in
  13.*)      IDX=cu130 ;;
  12.8|12.9) IDX=cu128 ;;
  12.6|12.7) IDX=cu126 ;;
  12.4|12.5) IDX=cu124 ;;
  *) die "CUDA $PICKV has no matching PyTorch build; use CUDA 12.4 or newer" ;;
esac
TORCH_SPEC=${TORCH_SPEC:-torch}
VENV=${EXL3_VENV:-$REPO/venv}
if command -v uv >/dev/null; then
  [ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 "$VENV"
  PIP=(uv pip install -p "$VENV/bin/python")
else
  PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3 || true)
  [ -n "$PY" ] || die "no python3 found (3.10 .. 3.13 supported)"
  "$PY" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 13)' 2>/dev/null || die "$PY is not Python 3.10 .. 3.13"
  [ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV" || die "python venv module missing (Debian/Ubuntu: apt install python3-venv)"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  PIP=("$VENV/bin/python" -m pip install)
fi
say "venv: $VENV; installing $TORCH_SPEC ($IDX)"
"${PIP[@]}" "$TORCH_SPEC" --index-url "https://download.pytorch.org/whl/$IDX" >/dev/null
TCUDA=$("$VENV/bin/python" -c 'import torch; print(torch.version.cuda or "")')
[ "${TCUDA%%.*}" = "${PICKV%%.*}" ] || die "PyTorch was built for CUDA $TCUDA but the toolkit is $PICKV (majors must match)"
"${PIP[@]}" -q setuptools wheel ninja

# 4. Build for this GPU only. RAM bounds parallelism: each nvcc job can take ~3 GB
CORES=$(nproc); RAM_GB=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
JOBS=${MAX_JOBS:-$(( CORES < RAM_GB / 3 ? CORES : RAM_GB / 3 ))}; [ "$JOBS" -ge 1 ] || JOBS=1
LOG=$REPO/build.log
say "building the extension for compute $ARCH with $JOBS jobs (typically 10-30 min; log: $LOG)"
if ! TORCH_CUDA_ARCH_LIST=$ARCH MAX_JOBS=$JOBS "${PIP[@]}" --no-build-isolation -v "$REPO" > "$LOG" 2>&1; then
  grep -m5 -E "error|Error|fatal" "$LOG" >&2 || true
  die "build failed; the first errors are above, full log in $LOG"
fi

# JIT-compiled kernels (tests, extensions loaded at runtime) must use the same toolkit as the build, not whatever
# nvcc is first on PATH
grep -q "EXL3 install.sh" "$VENV/bin/activate" || printf '\n# EXL3 install.sh: CUDA toolkit the extension was built with\nexport CUDA_HOME="%s"\nexport PATH="$CUDA_HOME/bin:$PATH"\n' "$CUDA_HOME" >> "$VENV/bin/activate"

# 5. Self-test
[ -n "${EXL3_NO_SELFTEST:-}" ] && { say "built; self-test skipped (EXL3_NO_SELFTEST). Run: $VENV/bin/python -m exllamav3.util.selftest"; exit 0; }
say "self-test"
"$VENV/bin/python" -m exllamav3.util.selftest || die "self-test failed (see above)"
say "done. Activate with: source $VENV/bin/activate"
