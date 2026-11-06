#!/usr/bin/env python3
# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""The benchmark inputs: whole causal prefills at 128K, 192K, 256K, 512K, 768K and 1M keys, BF16 and FP8.

usage: get_inputs.py random            the random set, generated here with seed 0
       get_inputs.py hf [--quick]      the trained (glm51) and untrained (init) sets as published (their 1M prefills,
                                       the shorter lengths cut from them), then their FP8
       get_inputs.py fp8               FP8 files of the BF16 files present (after dump_sglang.py)
       get_inputs.py text [--pin]      the InfiniteBench document dump_sglang.py prefills, checked against its sha256

Files: inputs/full_bf16/SET_LEN/callN.pt holds q [L, 32, 128] and kv [L, 128] in BF16, the head weights [L, 32] and
row r's window [starts, ends) = [0, r + 1); inputs/full_fp8/SET_LEN/callN.pt the same in FP8, as the DeepGEMM call
takes it: q and kv in e4m3 with power-of-two scales per query head and per key, the query scales folded into the
weights (w_eff). N is the layer for the model sets (24 and 48) and the seed for the random set. A published 1M file
is 9 GB; --quick takes layer 24 only (18 GB for both sets) and writes 128K and 1M.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import tomllib

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "inputs")
PINS = tomllib.load(open(os.path.join(HERE, "pins.toml"), "rb"))
LENGTHS = {"128k": 131072, "192k": 196608, "256k": 262144, "512k": 524288, "768k": 786432, "1m": 1048576}
SLICE = 65536


def pow2(amax):
    return torch.exp2(torch.ceil(torch.log2(amax.clamp(min=1e-30) / 448.0)))


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def to_fp8(bf16_path):
    """The FP8 file of a BF16 one: e4m3 with power-of-two scales per key and per query head."""
    fp8_path = bf16_path.replace(f"{os.sep}full_bf16{os.sep}", f"{os.sep}full_fp8{os.sep}")
    if os.path.exists(fp8_path):
        return
    d = torch.load(bf16_path, map_location="cpu", weights_only=False)
    q, kv, w, L = d["q"], d["kv"], d["weights"].float(), d["q"].shape[0]
    s = pow2(kv.float().abs().amax(-1))
    q_fp8 = torch.empty(L, 32, 128, dtype=torch.float8_e4m3fn)
    w_eff = torch.empty(L, 32, dtype=torch.float32)
    for a in range(0, L, SLICE):
        qa = q[a:a + SLICE].float()
        sq = pow2(qa.abs().amax(-1))
        q_fp8[a:a + SLICE] = (qa / sq[..., None]).to(torch.float8_e4m3fn)
        w_eff[a:a + SLICE] = w[a:a + SLICE] * sq
    save(dict(q_fp8=q_fp8, kv_fp8=(kv.float() / s[:, None]).to(torch.float8_e4m3fn).contiguous(),
              kv_scales=s.contiguous(), w_eff=w_eff, starts=d["starts"], ends=d["ends"],
              meta=dict(d.get("meta", {}), quant="pow2 per key / per query head")), fp8_path)
    print("fp8", os.path.relpath(fp8_path, HERE), flush=True)


def causal(q, kv, w, meta):
    L = q.shape[0]
    return dict(q=q, kv=kv, weights=w, starts=torch.zeros(L, dtype=torch.int32),
                ends=torch.arange(1, L + 1, dtype=torch.int32), meta=dict(meta, rows=L, nk=L))


def random_set(lengths, seed=0):
    """q, k ~ N(0, 0.5^2) in BF16 and signed head weights ~ N(0, 0.03^2), from the CPU generator."""
    for ctx in lengths:
        L = LENGTHS[ctx]
        path = os.path.join(INPUTS, "full_bf16", f"rand_{ctx}", f"call{seed}.pt")
        if not os.path.exists(path):
            g = torch.Generator().manual_seed(1000 * seed + L // 65536)
            q = torch.empty(L, 32, 128, dtype=torch.bfloat16)
            for a in range(0, L, SLICE):
                q[a:a + SLICE] = torch.randn(min(SLICE, L - a), 32, 128, generator=g) * 0.5
            kv = (torch.randn(L, 128, generator=g) * 0.5).bfloat16()
            w = torch.randn(L, 32, generator=g) * 0.03
            save(causal(q, kv, w, dict(source=f"random seed {seed}: q, k ~ N(0, 0.5^2), w ~ N(0, 0.03^2)")), path)
            print("bf16", os.path.relpath(path, HERE), flush=True)
        to_fp8(path)


def from_hf(quick):
    """The published 1M prefills, then every shorter length cut from them (each length is the run's prefix)."""
    from huggingface_hub import snapshot_download

    repo, revision = PINS["inputs"]["dataset"], PINS["inputs"]["revision"]
    if not repo:
        raise SystemExit("pins.toml names no published dataset yet: dump the inputs with dump_sglang.py")
    calls = "call24.pt" if quick else "call*.pt"
    snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=INPUTS,
                      allow_patterns=[f"full_bf16/*_1m/{calls}", "SHA256SUMS"])
    sums = os.path.join(INPUTS, "SHA256SUMS")
    if os.path.exists(sums):
        for line in open(sums):
            digest, name = line.split()
            path = os.path.join(INPUTS, name)
            if os.path.exists(path) and sha256(path) != digest:
                raise SystemExit(f"{name}: sha256 differs from the published one")
    for path in sorted(glob.glob(os.path.join(INPUTS, "full_bf16", "*_1m", calls))):
        prefixes(path, ["128k"] if quick else [c for c in LENGTHS if c != "1m"])
    fp8_all()


def prefixes(path, lengths):
    """BF16 call files of the first L tokens of a 1M one, for each length."""
    d = None
    for ctx in lengths:
        out = path.replace(f"_1m{os.sep}", f"_{ctx}{os.sep}")
        if os.path.exists(out):
            continue
        if d is None:
            d = torch.load(path, map_location="cpu", weights_only=False)
        L = LENGTHS[ctx]
        save(causal(d["q"][:L].clone(), d["kv"][:L].clone(), d["weights"][:L].clone(), d.get("meta", {})), out)
        print("bf16", os.path.relpath(out, HERE), flush=True)


def fp8_all():
    for path in sorted(glob.glob(os.path.join(INPUTS, "full_bf16", "*", "call*.pt"))):
        to_fp8(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def text(pin):
    """The document's context as UTF-8 text in text/, checked against pins.toml (with --pin, printed instead)."""
    from huggingface_hub import hf_hub_download

    t = PINS["text"]
    path = os.path.join(HERE, "text", f"{t['file'].split('.')[0]}_{t['id']}.txt")
    if not os.path.exists(path):
        src = hf_hub_download(t["dataset"], t["file"], repo_type="dataset", revision=t["revision"])
        doc = next(r for r in map(json.loads, open(src, encoding="utf-8")) if r["id"] == t["id"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w", encoding="utf-8").write(doc["context"])
    digest = sha256(path)
    if pin:
        print(f"sha256 = \"{digest}\"  # {os.path.relpath(path, HERE)}, {os.path.getsize(path)} bytes")
    elif digest != t["sha256"]:
        raise SystemExit(f"{path}: sha256 {digest}, pins.toml has {t['sha256']}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("random", "hf", "fp8", "text"))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--pin", action="store_true")
    p.add_argument("--lengths", default=",".join(LENGTHS))
    a = p.parse_args()
    if a.what == "random":
        random_set(["128k", "1m"] if a.quick else a.lengths.split(","))
    elif a.what == "hf":
        from_hf(a.quick)
    elif a.what == "fp8":
        fp8_all()
    else:
        print(text(a.pin))


if __name__ == "__main__":
    sys.exit(main())
