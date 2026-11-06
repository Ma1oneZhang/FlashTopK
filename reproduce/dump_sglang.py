#!/usr/bin/env python3
# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Regenerate the trained (glm51) and untrained (init) inputs: GLM-5.1's indexer inputs at layers 24 and 48 over the
first 1M tokens of the InfiniteBench document, from SGLang serving the model.

usage (the dump environment, `bash setup.sh dump`):
  .venv-dump/bin/python dump_sglang.py glm51 [--model PATH]   GLM-5.1 (BF16) at pins.toml's revision
  .venv-dump/bin/python dump_sglang.py init  [--model PATH]   the same model, weights drawn as Megatron-LM
                                                                initializes them (no checkpoint download)
The layers it runs take 905 GB of weights: on four nodes of 8 H800, the same command on each with
  --pp 4 --nnodes 4 --node-rank {0..3} --dist-init-addr HOST_OF_NODE_0:PORT

Only the layers up to the last dumped one run (the model is cut after layer 48: the earlier layers' outputs are the
same). One request prefills the document's first 1,048,576 tokens (TP8, chunked prefill, no CUDA graphs, the indexer's
unfused path). dump_hook/sitecustomize.py saves each chunk's BF16 q, k and head weights of the two layers; they are
then joined in position order into inputs/full_bf16/SET_LEN/callLAYER.pt for every length, each the first LEN tokens:
under causal attention a prefix's indexer inputs are the whole run's first rows (up to floating-point summation
order). The FP8 files follow (get_inputs.to_fp8). The model goes past its 202,752-token context, as the
measurements did; positions beyond it take the extrapolated RoPE.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import get_inputs  # noqa: E402

PINS = tomllib.load(open(os.path.join(HERE, "pins.toml"), "rb"))


def prefill(model, revision, ids, init, raw, a):
    os.environ.update(FLASHTOPK_DUMP_DIR=raw, FLASHTOPK_DUMP_LAYERS=",".join(map(str, PINS["dump"]["layers"])),
                      SGLANG_DISABLE_DSA_INDEXER_FUSION="1", SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="1",
                      PYTHONPATH=os.pathsep.join(filter(None, [os.path.join(HERE, "dump_hook"),
                                                               os.environ.get("PYTHONPATH")])))
    if init:
        from transformers import AutoConfig

        # The Megatron-LM settings of the released measurements' untrained dumps: --init-method-std 6e-3, and the
        # output projections scaled by the depth of the whole model.
        layers = AutoConfig.from_pretrained(model, revision=revision).num_hidden_layers
        os.environ.update(FLASHTOPK_INIT="megatron", FLASHTOPK_INIT_STD="0.006", FLASHTOPK_INIT_LAYERS=str(layers))
    sys.path.insert(0, os.path.join(HERE, "dump_hook"))
    import sitecustomize  # noqa: F401  (this process too)
    import sglang

    # Use the measured SM90 attention, MoE and GEMM backends, with an FP8 KV cache.
    engine = sglang.Engine(model_path=model, revision=revision, tp_size=a.tp, pp_size=a.pp, nnodes=a.nnodes,
                           node_rank=a.node_rank, dist_init_addr=a.dist_init_addr, context_length=a.tokens + 64,
                           json_model_override_args=json.dumps({"num_hidden_layers": max(PINS["dump"]["layers"]) + 1}),
                           chunked_prefill_size=a.chunk, max_prefill_tokens=a.chunk, disable_cuda_graph=True,
                           disable_prefill_cuda_graph=True, disable_decode_cuda_graph=True, disable_radix_cache=True,
                           kv_cache_dtype="fp8_e4m3", dsa_prefill_backend="flashmla_kv", dsa_decode_backend="flashmla_kv",
                           moe_runner_backend="triton", fp8_gemm_runner_backend="triton",
                           # Every pipeline stage compiles its kernels on the first chunk, for minutes; the later
                           # stages wait meanwhile, past the 300 s of SGLang's watchdog.
                           watchdog_timeout=3600,
                           mem_fraction_static=a.mem_fraction, load_format="dummy" if init else "auto", random_seed=1)
    try:
        engine.generate(input_ids=ids, sampling_params=dict(max_new_tokens=1, temperature=0))
    finally:
        engine.shutdown()


def assemble(raw, name, meta, tokens):
    """Join a layer's chunks in position order and write every length's prefix as a BF16 call file."""
    import torch

    for layer in PINS["dump"]["layers"]:
        parts = sorted(glob.glob(os.path.join(raw, f"layer{layer}", "*.pt")))
        chunks = [torch.load(p) for p in parts]
        start = max(i for i, c in enumerate(chunks) if int(c["positions"][0]) == 0)  # the last request
        chunks = chunks[start:]
        pos = torch.cat([c["positions"] for c in chunks]).long()
        if not torch.equal(pos, torch.arange(pos.numel())) or pos.numel() < tokens:
            raise SystemExit(f"layer {layer}: the chunks cover positions 0..{pos.numel() - 1} with gaps or too few")
        q = torch.cat([c["q"] for c in chunks])[:tokens].contiguous()
        k = torch.cat([c["k"] for c in chunks])[:tokens].contiguous()
        w = torch.cat([c["weights"] for c in chunks])[:tokens].contiguous()
        for ctx, L in ((c, L) for c, L in get_inputs.LENGTHS.items() if L <= tokens):
            path = os.path.join(get_inputs.INPUTS, "full_bf16", f"{name}_{ctx}", f"call{layer}.pt")
            get_inputs.save(get_inputs.causal(q[:L].clone(), k[:L].clone(), w[:L].clone(), dict(meta, layer=layer)),
                            path)
            print("bf16", os.path.relpath(path, HERE), flush=True)
            get_inputs.to_fp8(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("set", choices=("glm51", "init"))
    p.add_argument("--model", default=PINS["dump"]["model"], help="HF repo or a local copy of the checkpoint")
    p.add_argument("--chunk", type=int, default=16384, help="SGLang's chunked_prefill_size")
    p.add_argument("--tokens", type=int, default=1 << 20, help="tokens to prefill; the lengths up to it are written")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--pp", type=int, default=1, help="pipeline stages, one per node with --nnodes")
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--dist-init-addr", help="HOST:PORT of node 0, with --nnodes")
    p.add_argument("--mem-fraction", type=float, default=0.6,
                   help="SGLang's mem_fraction_static: weights and KV cache; the rest serves the indexer's 1M-key logits")
    p.add_argument("--keep-raw", action="store_true")
    a = p.parse_args()
    from transformers import AutoTokenizer

    revision = PINS["dump"]["model_revision"] if a.model == PINS["dump"]["model"] else None
    raw = os.path.join(get_inputs.INPUTS, "raw", a.set)
    if a.node_rank:  # SGLang keeps the other nodes serving node 0's request; they save their layers' chunks too
        prefill(a.model, revision, None, a.set == "init", raw, a)
        return
    text = open(get_inputs.text(pin=False), encoding="utf-8").read()
    ids = AutoTokenizer.from_pretrained(a.model, revision=revision).encode(text, add_special_tokens=False)
    if len(ids) < a.tokens:
        raise SystemExit(f"the document is {len(ids)} tokens, under {a.tokens}")
    shutil.rmtree(raw, ignore_errors=True)
    prefill(a.model, revision, ids[:a.tokens], a.set == "init", raw, a)
    import sglang

    meta = dict(source=f"{PINS['text']['dataset']}@{PINS['text']['revision']} {PINS['text']['file']} "
                       f"id {PINS['text']['id']}, first {a.tokens} tokens",
                model=f"{PINS['dump']['model']}@{PINS['dump']['model_revision']}",
                init="megatron" if a.set == "init" else "checkpoint",
                sglang=sglang.__version__, chunk=a.chunk)
    assemble(raw, a.set, meta, a.tokens)
    if not a.keep_raw:
        shutil.rmtree(raw)


if __name__ == "__main__":
    main()
