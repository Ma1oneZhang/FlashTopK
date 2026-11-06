# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""The methods of the full-prefill comparison, each splitting a prefill as it does on its own.

A call is one causal prefill of L keys: row r of q sees keys [0, r + 1). FP8 operands are q_fp8 [L, 32, 128],
kv_fp8 [L, 128], kv_scales [L] and w_eff [L, 32]; BF16 ones q_bf16, kv_bf16 and w (rounded to BF16 values, as cuDNN
takes them). Each method returns the [L, TOPK] int32 indices.

  flashtopk   FlashTopK, chunk_size rows at a time (its default 65536)
  dense       the dense table, DeepGEMM fp8_fp4_mqa_logits (FP8) or cuDNN's indexer forward (BF16), then FlashInfer's
              ragged top-k: row blocks whose FP32 table fits in 35% of free memory, in whole 4096-row blocks
  hpc_topk    the same table, hpc.topk (HPC-Ops)
  deepselect  the same table, deep_select.topk (indices only, unsorted)
"""

import ctypes
import hashlib
import os
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

HERE = os.path.dirname(os.path.abspath(__file__))
TOPK = 2048
ROW_KEYS = ("q_fp8", "w_eff", "q_bf16", "w", "w_bf16", "starts", "ends")
DENSE_MEMORY_SHARE = 0.75  # the table is built whole while it takes at most this share of the GPU's memory
DENSE_FREE_SHARE = 0.35    # and at most this share of what is free when the call starts


def is_bf16(op):
    return "q_bf16" in op


def q_of(op):
    return op["q_bf16"] if is_bf16(op) else op["q_fp8"]


def nk_of(op):
    return (op["kv_bf16"] if is_bf16(op) else op["kv_fp8"]).shape[0]


def causal_rows(op, s):
    """Rows s of a prefill with the keys they reach, kv[:s.stop]: a causal call of its own."""
    sub = {k: op[k][s] for k in ROW_KEYS if k in op}
    if is_bf16(op):
        sub["kv_bf16"] = op["kv_bf16"][:s.stop]
    else:
        sub.update(kv_fp8=op["kv_fp8"][:s.stop], kv_scales=op["kv_scales"][:s.stop])
    return sub


def cudnn_logits(op, s):
    """cuDNN's FP32 scores of rows s over the keys their causal windows reach. Its mask is bottom-right causal, so
    the keys stop at the slice's last window."""
    from cudnn.deepseek_sparse_attention.indexer_forward.api import indexer_forward_wrapper

    end = nk_of(op) - q_of(op).shape[0] + s.stop
    kv = op["kv_bf16"][:end]
    return indexer_forward_wrapper(op["q_bf16"][s][None], kv[None, :, None, :], op["w_bf16"][s][None],
                                   ratio=1)["scores"][0]


def dense_logits(op, rows):
    """The dense FP32 scores of rows (a slice), -inf outside each row's window: the reference of the checks."""
    nk = nk_of(op)
    if is_bf16(op):
        part = cudnn_logits(op, rows)
        out = torch.full((part.shape[0], nk), float("-inf"), dtype=torch.float32, device=part.device)
        width = min(part.shape[1], nk)
        out[:, :width] = part[:, :width]
        cols = torch.arange(nk, device=part.device)
        inside = (cols[None] >= op["starts"][rows, None].long()) & (cols[None] < op["ends"][rows, None].long())
        return out.masked_fill_(~inside, float("-inf"))
    import deep_gemm

    return deep_gemm.fp8_fp4_mqa_logits((op["q_fp8"][rows], None), (op["kv_fp8"], op["kv_scales"]), op["w_eff"][rows],
                                        op["starts"][rows], op["ends"][rows], clean_logits=True)[:, :nk]


def host_bytes(t):
    """A tensor's bytes in memory order on the host (numpy's tobytes, without numpy)."""
    t = t.contiguous().cpu()
    return ctypes.string_at(t.data_ptr(), t.numel() * t.element_size())


class _RowSlack(TorchDispatchMode):
    """One spare row for DeepGEMM's [rows, padded] FP32 logits: it pads the row stride but not the last row, so with
    the spare row the padded table reads as one contiguous matrix, each row bounded by its length."""

    def __init__(self):
        super().__init__()
        self.padded = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (func is torch.ops.aten.empty.memory_format and kwargs.get("dtype") == torch.float32
                and len(args[0]) == 2 and self.padded is None):
            rows, cols = args[0]
            self.padded = torch.empty((rows + 1) * cols, dtype=torch.float32, device=kwargs.get("device"))[
                : rows * cols].view(rows, cols)
            return self.padded
        return func(*args, **kwargs)


class Dense:
    """The dense table and FlashInfer's ragged top-k, a row block at a time. Subclasses keep the scoring and replace
    select() with another top-k over the same table."""

    def __init__(self, args):
        import flashinfer

        self.fi = flashinfer
        self.total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        self.blocks = {}

    def score(self, op, s):
        if is_bf16(op):
            return cudnn_logits(op, s)
        import deep_gemm

        mode = _RowSlack()
        with mode:
            logits = deep_gemm.fp8_fp4_mqa_logits((op["q_fp8"][s], None), (op["kv_fp8"], op["kv_scales"]),
                                                  op["w_eff"][s], op["starts"][s], op["ends"][s], clean_logits=False)
        if mode.padded is None or logits.data_ptr() != mode.padded.data_ptr():
            raise RuntimeError("DeepGEMM did not return the reserved padded logits")
        return mode.padded

    def select(self, op, s, logits):
        return self.fi.top_k_ragged_transform(logits, op["starts"][s], op["ends"][s] - op["starts"][s], TOPK,
                                              deterministic=False, dsa_graph_safe=True)

    def block_rows(self, rows, nk):
        """Rows per block, decided once per shape so the timed calls all use the same one."""
        if (rows, nk) not in self.blocks:
            budget = min(DENSE_MEMORY_SHARE * self.total, DENSE_FREE_SHARE * torch.cuda.mem_get_info()[0])
            block = max(1, min(rows, int(budget // (4 * nk))))
            if block < rows:  # whole 4096-row blocks: DeepGEMM's logits fail on some ragged row counts
                block = max(min(4096, block), block // 4096 * 4096)
            self.blocks[rows, nk] = block
        return self.blocks[rows, nk]

    def __call__(self, op):
        rows, nk = q_of(op).shape[0], nk_of(op)
        block = self.block_rows(rows, nk)
        self.last_block, self.call_rows = block, rows
        if block >= rows:
            return self.select(op, slice(0, rows), self.score(op, slice(0, rows)))
        return torch.cat([self.select(op, s, self.score(op, s))
                          for s in (slice(r, min(rows, r + block)) for r in range(0, rows, block))])

    def stats(self, op):
        return dict(row_block=getattr(self, "last_block", None))


def _from_third_party(name):
    path = os.path.join(HERE, "third_party", "envs", name)
    if not os.path.isdir(path):
        raise RuntimeError(f"{path} missing: run setup.sh")
    sys.path.insert(0, path)


class HPCOpsTopK(Dense):
    """hpc.topk on the dense table, with the workspace hpc.topk_workspace_size recommends, allocated once."""

    def __init__(self, args):
        super().__init__(args)
        _from_third_party("hpc-ops")
        import hpc

        self.hpc = hpc
        self.buffers, self.out = {}, {}

    def select(self, op, s, logits):
        rows, cols = logits.shape[0], max(logits.shape[1], nk_of(op))
        if (rows, cols) not in self.buffers:
            counters, workspace = self.hpc.topk_workspace_size(rows, cols)
            self.buffers[rows, cols] = dict(
                live=torch.tensor([rows], dtype=torch.int32, device="cuda"),
                counters=torch.zeros(counters, dtype=torch.uint8, device="cuda"),
                workspace=torch.empty(workspace, dtype=torch.uint8, device="cuda"))
        b = self.buffers[rows, cols]
        if self.call_rows not in self.out:  # the call's whole output, so blocks do not overwrite each other
            self.out[self.call_rows] = torch.empty(self.call_rows, TOPK, dtype=torch.int32, device="cuda")
        out = self.out[self.call_rows][s]
        self.hpc.topk(logits, op["ends"][s].contiguous(), out, b["live"], top_k=TOPK, counters=b["counters"],
                      workspace=b["workspace"])
        return out


class DeepSelectTopK(Dense):
    """deep_select.topk on the dense table, each row bounded by its window's end; indices only and unsorted, its
    fast path."""

    def __init__(self, args):
        super().__init__(args)
        _from_third_party("deepselect")
        import deep_select

        self.ds = deep_select
        self.out = {}

    def select(self, op, s, logits):
        if self.call_rows not in self.out:
            self.out[self.call_rows] = torch.empty(self.call_rows, TOPK, dtype=torch.int32, device="cuda")
        out = self.out[self.call_rows][s]
        self.ds.topk(logits, TOPK, end=op["ends"][s].contiguous(), indices_type=torch.int32, sorted_index=False,
                     return_value=False, output_idx=out)
        return out


class FlashTopK:
    """FlashTopK: one call over every row, run in slices of its chunk_size (FLASHTOPK_CHUNK, default 65536)."""

    def __init__(self, args):
        import flashtopk

        self.flashtopk = flashtopk
        self.chunk_size = int(os.environ["FLASHTOPK_CHUNK"]) if os.environ.get("FLASHTOPK_CHUNK") else None
        self.selector = self.scores = None

    def __call__(self, op):
        rows, nk = q_of(op).shape[0], nk_of(op)
        if self.selector is None:  # the scores serve the digest
            self.selector = self.flashtopk.Selector(TOPK, dtype=q_of(op).dtype, chunk_size=self.chunk_size,
                                                    return_scores=True)
        sel = self.selector.plan(rows, nk, warmup=False)
        if is_bf16(op):
            call = (op["q_bf16"], op["kv_bf16"], op["w"], op["starts"], op["ends"], None)
        else:
            call = (op["q_fp8"], op["kv_fp8"], op["w_eff"], op["starts"], op["ends"], op["kv_scales"])
        self.scores, indices = sel.run(*call)
        return indices

    def digest(self, indices, rows_at_once=65536):
        """SHA-256 of the selection sorted by index: the indices, then the scores' bits."""
        h = hashlib.sha256()
        for a in range(0, indices.shape[0], rows_at_once):
            h.update(host_bytes(indices[a:a + rows_at_once].sort(dim=-1)[0].int()))
        for a in range(0, indices.shape[0], rows_at_once):
            order = indices[a:a + rows_at_once].sort(dim=-1)[1]
            h.update(host_bytes(self.scores[a:a + rows_at_once].gather(1, order).view(torch.int32)))
        return h.hexdigest()

    def stats(self, op):
        info = self.selector.last_info
        health = info.health.tolist()
        return dict(pruned=bool(info.pruned), overflow_rows=health[0], underfill_rows=health[1], bad_rows=health[7],
                    chunk_size=self.selector.chunk_size, version=self.flashtopk.__version__)


METHODS = dict(flashtopk=FlashTopK, dense=Dense, hpc_topk=HPCOpsTopK, deepselect=DeepSelectTopK)
