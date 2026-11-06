#!/usr/bin/env bash
# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
#
# The environments of the reproduction, at the versions of pins.toml.
#
#   bash setup.sh          the benchmark environment (.venv): the pinned packages and FlashTopK, the baselines' sources
#                          at their pinned commits in third_party/, DeepSelect and HPC-Ops built for SM90
#   bash setup.sh dump     the dump environment (.venv-dump): SGLang for dump_sglang.py, only to regenerate the inputs
#
# PYTHON picks the interpreter (default python3; 3.11 to 3.14, and to 3.13 for dump, SGLang's wheels), and SOURCES,
# on a machine without GitHub, a directory of git checkouts at the pinned commits (one per name of pins.toml)
# to copy instead. ARCH defaults to 90; only SM90 builds are supported.
# Building DeepSelect takes most of the time (about half an hour with 32 cores). Rerunning skips what is done.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
ARCH=${ARCH:-90}
[ "$ARCH" = 90 ] || { echo "ARCH must be 90 (Hopper), got '$ARCH'"; exit 1; }
export MAX_JOBS=${MAX_JOBS:-32} NVCC_THREADS=${NVCC_THREADS:-8}

pins() { "$1" -c "import sys, tomllib; t = tomllib.load(open('pins.toml', 'rb')); exec(sys.argv[1])" "$2"; }

if [ "${1:-}" = dump ]; then
    if [ ! -x .venv-dump/bin/python ]; then
        "$PY" -c "import os, sysconfig; assert os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h'))" ||
            { echo "$PY has no Python.h, which SGLang's Triton compiles against: install its -dev / -devel package"; exit 1; }
        "$PY" -m venv .venv-dump
    fi
    .venv-dump/bin/python -m pip install -q -U pip
    .venv-dump/bin/python -m pip install -q "sglang==$(pins .venv-dump/bin/python "print(t['dump']['sglang'])")" \
        "huggingface-hub==$(pins .venv-dump/bin/python "print(t['dump']['huggingface-hub'])")"
    echo "SETUP_DUMP_DONE: $(.venv-dump/bin/python -c 'import sglang; print("sglang", sglang.__version__)')"
    exit 0
fi

[ -x .venv/bin/python ] || "$PY" -m venv .venv
P=$PWD/.venv/bin/python
pins "$P" "print('\n'.join(f'{k}=={v}' for k, v in t['pip'].items()))" > requirements.txt
"$P" -m pip install -q -U pip
"$P" -m pip install -q -r requirements.txt
"$P" -m pip install -q --no-deps $(pins "$P" "print(' '.join(f'{k}=={v}' for k, v in t['pip-nodeps'].items()))")
"$P" -m pip install -q --no-deps -e ..

mkdir -p third_party/envs
pins "$P" "[print(n, g['url'], g['commit']) for n, g in t['git'].items()]" |
while read -r name url commit; do
    d=third_party/$name
    [ "$(git -C "$d" rev-parse HEAD 2>/dev/null || true)" = "$commit" ] && continue
    rm -rf "$d"
    if [ -n "${SOURCES:-}" ]; then  # a checkout at the commit, made where GitHub is reachable
        cp -a "$SOURCES/$name" "$d"
    else
        git init -q "$d"
        git -C "$d" fetch -q --depth 1 "$url" "$commit"
        git -C "$d" checkout -q FETCH_HEAD
    fi
    [ "$(git -C "$d" rev-parse HEAD)" = "$commit" ] || { echo "$name: not at $commit"; exit 1; }
    echo "$name at $commit"
done
CUTLASS=$PWD/third_party/cutlass

build() {  # NAME MODULE: once the built package imports, link it into third_party/envs/NAME, where bench.py looks for it
    local lib=(third_party/"$1"/build/lib*/)
    [ -d "$lib" ] && PYTHONPATH=$lib "$P" -c "import torch, $2" ||
        { echo "$1: build failed, see third_party/$1.build.log; rm -rf third_party/$1/build to rebuild"; exit 1; }
    ln -sfn "$PWD/${lib%/}" "third_party/envs/$1"
}

# HPC-Ops: only its top-k operator (its CMake compiles every operator it finds), against CUTLASS 4.4.2.
if [ ! -e third_party/envs/hpc-ops ]; then
    (
        cd third_party/hpc-ops
        rm -rf 3rd/cutlass && ln -sfn "$CUTLASS" 3rd/cutlass
        for d in src/*/; do case $(basename "$d") in topk|C|utils) ;; *) rm -rf "$d" ;; esac; done
        SM_ARCH=$ARCH "$P" setup.py build > ../hpc-ops.build.log 2>&1
        # hpc/__init__.py imports every operator's module, and those this build left out fail to register.
        for f in build/lib*/hpc/*.py; do
            case $(basename "$f") in __init__.py|version.py|topk.py) ;; *) rm -f "$f" ;; esac
        done
    )
    build hpc-ops hpc
fi

# DeepSelect, against CUTLASS 4.4.2, compiled only for sm_90a. Its std::format
# needs GCC 13+, and a gcc-toolset that new leaves symbols to a libstdc++ (GLIBCXX_3.4.31+) older systems lack, so
# libstdc++ is linked in statically.
if [ ! -e third_party/envs/deepselect ]; then
    (
        cd third_party/deepselect
        rm -rf csrc/3rdparty/cutlass && ln -sfn "$CUTLASS" csrc/3rdparty/cutlass
        "$P" - <<'EOF'
from pathlib import Path
import re

path = Path("setup.py")
source, count = re.subn(r"    cc_flag = \[[^\]]*\]",
                        '    cc_flag = ["-gencode", "arch=compute_90a,code=sm_90a"]',
                        path.read_text(), count=1)
if count != 1:
    raise SystemExit("DeepSelect setup.py: expected one cc_flag list")
path.write_text(source)
EOF
        LDFLAGS="${LDFLAGS:-} -static-libstdc++" DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA \
            DEEP_SELECT_DISABLE_REG_SPILL_CHECK=1 "$P" setup.py build > ../deepselect.build.log 2>&1
    )
    build deepselect deep_select
fi

"$P" - <<'EOF'
import importlib, sys
sys.path[:0] = ["third_party/envs/hpc-ops", "third_party/envs/deepselect"]
for m in ("flashtopk", "flashinfer", "deep_gemm", "cudnn", "hpc", "deep_select"):
    importlib.import_module(m)
print("SETUP_DONE: imports ok")
EOF
