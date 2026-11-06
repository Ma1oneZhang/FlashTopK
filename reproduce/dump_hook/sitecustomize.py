# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""dump_sglang.py's hooks, installed in every process SGLang spawns (this directory is on their PYTHONPATH) when
FLASHTOPK_DUMP_DIR is set.

* The DSA indexer's _get_q_k_bf16: for the layers in FLASHTOPK_DUMP_LAYERS, TP rank 0 of the layer's pipeline stage
  saves each prefill chunk's BF16 q [n, 32, 128] and k [n, 128] as the indexer scores them (after the Hadamard
  rotation), and the head weights its logits take without the FP8 query scales,
  weights_proj(x) * n_heads^-0.5 * softmax_scale, to FLASHTOPK_DUMP_DIR/layer<L>/<first position>.pt.
* With FLASHTOPK_INIT=megatron, the dummy weights are drawn as Megatron-LM initializes a model, in place of SGLang's
  uniform(-1e-3, 1e-3) for every tensor: N(0, s^2) matrices with s = FLASHTOPK_INIT_STD, and for the attention and MLP
  output projections s / sqrt(2 * FLASHTOPK_INIT_LAYERS); unit norm weights and FP8 block scales, zero biases; each
  tensor (each TP shard of a split one) from its own seed.
"""
import importlib.abc
import importlib.util
import os
import sys


def _patch_indexer(module):
    import torch
    from sglang.srt.distributed import get_tensor_model_parallel_rank

    layers = {int(x) for x in os.environ["FLASHTOPK_DUMP_LAYERS"].split(",")}
    root = os.environ["FLASHTOPK_DUMP_DIR"]
    original = module.Indexer._get_q_k_bf16

    def _get_q_k_bf16(self, q_lora, x, positions, enable_dual_stream, forward_batch):
        query, key, weights_raw = original(self, q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch)
        # TP rank 0 of each pipeline stage saves the layers of that stage.
        tp0 = get_tensor_model_parallel_rank() == 0
        if self.layer_id in layers and tp0 and forward_batch.forward_mode.is_extend() and query.shape[0]:
            if weights_raw is None:
                if isinstance(x, tuple):  # (fp8, scale, bf16) from a fused RMSNorm + quant: the BF16 copy
                    if len(x) != 3:
                        raise RuntimeError("an FP8-only indexer input: set the dump's BF16 path")
                    x = x[2]
                weights_raw = torch.mm(x, self.weights_proj.weight.t(), out_dtype=torch.float32)
            weights = weights_raw.float() * self.n_heads**-0.5 * self.softmax_scale
            n = query.shape[0]
            path = os.path.join(root, f"layer{self.layer_id}", f"{int(positions[0]):08d}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(dict(q=query.detach().cpu(), k=key[-n:].detach().cpu(), weights=weights.cpu(),
                            positions=positions.cpu()), path)
        return query, key, weights_raw

    module.Indexer._get_q_k_bf16 = _get_q_k_bf16


def _patch_loader(module):
    import math
    import zlib

    import torch
    from sglang.srt.distributed import get_tensor_model_parallel_rank
    from sglang.srt.layers.linear import ReplicatedLinear

    def initialize_dummy_weights(model, low=-1e-3, high=1e-3, seed=1234):
        # A tensor split over the TP ranks draws each shard from a seed of its own, as Megatron-LM does; a replicated
        # one draws the same on every rank.
        split = set()
        for prefix, mod in model.named_modules():
            if not isinstance(mod, ReplicatedLinear) and \
                    max(getattr(mod, a, None) or 1 for a in ("tp_size", "moe_tp_size", "moe_ep_size")) > 1:
                split.update(f"{prefix}.{n}" if prefix else n for n, _ in mod.named_parameters(recurse=False))
        rank = get_tensor_model_parallel_rank()
        std = float(os.environ["FLASHTOPK_INIT_STD"])
        out_std = std / math.sqrt(2 * int(os.environ["FLASHTOPK_INIT_LAYERS"]))  # the output projections, scaled
        for name, param in model.state_dict().items():
            if not torch.is_floating_point(param):
                continue
            g = torch.Generator(device=param.device)
            g.manual_seed(seed + zlib.crc32(name.encode()) + ((rank + 1) << 32 if name in split else 0))
            leaf = name.rsplit(".", 1)[-1]
            if leaf.endswith("bias"):
                param.data.zero_()
            elif "scale" in leaf or (param.dim() == 1 and "norm" in name):
                param.data.fill_(1.0)
            elif param.dim() == 1:
                param.data.zero_()
            else:
                s = out_std if any(k in name for k in (".o_proj.", ".down_proj.", ".w2_weight")) else std
                draw = torch.empty(param.shape, dtype=torch.float32, device=param.device).normal_(0, s, generator=g)
                param.data.copy_(draw.to(param.dtype))

    module.initialize_dummy_weights = initialize_dummy_weights


class _AfterImport(importlib.abc.MetaPathFinder):
    """Run a patch on a module right after it executes."""

    def __init__(self, patches):
        self.patches = patches

    def find_spec(self, name, path, target=None):
        if name not in self.patches:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        patch, run = self.patches.pop(name), spec.loader.exec_module

        def exec_module(module):
            run(module)
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("FLASHTOPK_DUMP_DIR"):
    patches = {"sglang.srt.layers.attention.dsa.dsa_indexer": _patch_indexer}
    if os.environ.get("FLASHTOPK_INIT") == "megatron":
        patches["sglang.srt.model_loader.loader"] = _patch_loader
    sys.meta_path.insert(0, _AfterImport(patches))
