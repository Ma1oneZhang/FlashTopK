#!/usr/bin/env bash
# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
#
# Every method over the inputs on one GPU, one process per method and input set (each method's JIT builds and memory
# stay its own), into results/<card>_<dtype>/<set>/<method>.jsonl. A rerun resumes after the last recorded call.
#
# usage: [GPU=0] [SETS="glm init random"] [TYPES="fp8 bf16"] [LENGTHS="128k 192k 256k 512k 768k 1m"]
#        [METHODS="dense flashtopk hpc_topk deepselect"] [CARD=h800] bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
P=$PWD/.venv/bin/python
GPU=${GPU:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONWARNINGS=ignore TORCH_EXTENSIONS_DIR=$PWD/.cache/torch_extensions
# The dense path allocates tens of GiB per block; without this, blocks of a new shape cannot reuse the segments.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SM=$("$P" -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability())))')
[ "$SM" = 9.0 ] || { echo "the reproduction benchmarks require an SM90 GPU, got capability $SM"; exit 1; }
CARD=${CARD:-$("$P" -c 'import torch; print(torch.cuda.get_device_name().split()[-1].lower())')}
METHODS=${METHODS:-"dense flashtopk hpc_topk deepselect"}
failed=0
ran=0
for m in $METHODS; do
    case $m in dense|flashtopk|hpc_topk|deepselect) ;; *) echo "unknown method $m"; exit 1 ;; esac
done
for L in ${LENGTHS:-128k 192k 256k 512k 768k 1m}; do
    case $L in 128k|192k|256k|512k|768k|1m) ;; *) echo "unknown length $L"; exit 1 ;; esac
done

for t in ${TYPES:-fp8 bf16}; do
    case $t in fp8|bf16) ;; *) echo "unknown operand type $t"; exit 1 ;; esac
    for set in ${SETS:-glm init random}; do
        case $set in glm) prefix=glm51 ;; init) prefix=init ;; random) prefix=rand ;; *) echo "unknown set $set"; exit 1 ;; esac
        calls=()
        for L in ${LENGTHS:-128k 192k 256k 512k 768k 1m}; do
            for f in inputs/full_$t/${prefix}_$L/call*.pt; do [ -e "$f" ] && calls+=("$f"); done
        done
        [ ${#calls[@]} -gt 0 ] || { echo "no $t $set inputs (get_inputs.py)"; continue; }
        out=results/${CARD}_$t/$set
        mkdir -p "$out"
        for m in $METHODS; do
            ran=1
            echo "[$(date +%T)] $CARD $t $set $m: ${#calls[@]} calls"
            # A failed process can resume from its recorded calls.
            for attempt in 1 2 3; do
                "$P" bench.py --method "$m" --out "$out/$m.jsonl" "${calls[@]}" >> "$out/$m.log" 2>&1 && break
                echo "[$(date +%T)] $m attempt $attempt failed, resuming"
                [ "$attempt" -lt 3 ] || failed=1
            done
        done
    done
done
[ "$ran" = 1 ] || { echo "RUN_FAILED: no inputs found; run get_inputs.py first"; exit 1; }
[ "$failed" = 0 ] || { echo "RUN_FAILED: see results/$CARD*/ for logs"; exit 1; }
echo "[$(date +%T)] RUN_DONE: .venv/bin/python report.py"
