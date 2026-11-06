# FlashTopK API

See the [README](../README.md) for the score, installation and a first call.

## Semantics

- **Exact:** rows the sampled threshold misses are rescored on the GPU (recovery), so every row is the exact top-k. Recovery is always on.
- **Deterministic:** ties at the k-th score keep the smallest key indices. The output is sorted by score, then by index.
- **Adaptive key pruning:** from 128K keys, each call decides on the GPU whether to skip keys by a score bound (see [Configuration](#configuration)).
- **Asynchronous:** a call never synchronizes with the host. Selection carries no gradient.

## flashtopk.select

```python
flashtopk.select(q, kv, weights, ks, ke, kv_scales=None, topk=2048, pruning=None, out=None, chunk_size=None,
                 return_scores=False) -> indices, or (scores, indices)
"""Each query row's top-k keys under the indexer score.
Keeps one Selector per (device, dtype, topk, pruning, chunk_size, return_scores); its workspace grows to the largest
call seen.

Arguments:
    q: (num_q, 32, 128), torch.float8_e4m3fn or torch.bfloat16. The indexer queries.
    kv: (seq_len_kv, 128), the dtype of q. The indexer keys, contiguous (no paged KV cache).
    weights: (num_q, 32), float32. The per-head weights of each query row.
    ks, ke: (num_q,), int32. Each row's window of keys [ks, ke).
    kv_scales: (seq_len_kv,), float32. The per-key scales of FP8 keys. None for BF16 keys.
    topk: int in [1, 4096]. The keys to select per row.
    pruning: "adaptive", "always" or "off". Default: $FLASHTOPK_PRUNING, else "adaptive".
    out: indices, or (scores, indices) with return_scores, optional. Buffers to write the result into.
    chunk_size: int. The query rows processed at once. A call with more rows, such as a whole prefill, runs in
        slices of chunk_size rows and returns the same bits. Default: flashtopk.DEFAULT_CHUNK_SIZE (65536).
    return_scores: bool. Whether to return the scores with the indices, num_q * topk * 4 bytes more. Default: False.
Return:
    indices: (num_q, topk), int32. Key indices into kv, by descending score; equal scores in ascending index order.
        A window shorter than topk is padded with -1.
    scores [optional, if return_scores=True]: (num_q, topk), float32. Descending; -0.0 ties +0.0; padded with -inf.
        Returned as (scores, indices).
"""
```

## flashtopk.Selector

A `Selector` owns its workspace. Use it to allocate memory and compile the kernels before the first call.

```python
selector = flashtopk.Selector(topk=2048, num_heads=32, dtype=torch.float8_e4m3fn, device=None, pruning=None,
                              chunk_size=None, return_scores=False)
"""topk, pruning, chunk_size, return_scores: as in flashtopk.select.
num_heads: must be 32. dtype: torch.float8_e4m3fn or torch.bfloat16. device: default the current CUDA device.
"""

selector.plan(max_rows, max_kv, warmup=True) -> Selector
"""Allocates the workspace for calls of up to max_rows query rows over max_kv keys: about
8 * min(max_rows, chunk_size) * max(4 * topk, 16384) bytes, plus a slice's scores (4 * min(max_rows, chunk_size) *
topk bytes) unless return_scores. flashtopk.workspace_bytes(max_rows, max_kv, topk, chunk_size, return_scores) gives it.
With warmup, compiles every CuTe kernel variant these calls can take.
Optional: run() grows the plan when a call is larger.
"""

selector.run(q, kv, weights, ks, ke, kv_scales=None, out=None) -> indices, or (scores, indices)
"""The arguments and outputs of flashtopk.select."""

selector.last_info
"""The last call's record. It stays on the GPU until read; reading synchronizes.
    pruned: bool. Whether the call pruned keys.
    bad_rows: int. The rows recovery rescored.
    health: int64[8]. [0] rows whose candidates overflowed, [1] rows that came up short, [7] bad rows.
"""
```

For example:

```python
selector = flashtopk.Selector(topk=2048, dtype=torch.float8_e4m3fn).plan(max_rows=8192, max_kv=1 << 20)
indices = selector.run(q, kv, weights, ks, ke, kv_scales)
```

## Other interfaces

```python
flashtopk.windows_from_cu_seqlens(cu_seqlens_q, cu_seqlens_k) -> (ks, ke)
"""Causal windows of packed sequences (varlen prefill), int32.
The queries of sequence i are the last of its keys. Query t of sequence i sees keys
[cu_seqlens_k[i], cu_seqlens_k[i] + (len_k - len_q) + t + 1).
"""

torch.ops.flashtopk.select(q, kv, weights, ks, ke, kv_scales, topk, pruning=None, chunk_size=None,
                           return_scores=False) -> (scores, indices)
"""flashtopk.select as a custom op, so torch.compile traces through it. scores is empty, (num_q, 0), unless
return_scores."""

flashtopk.supported(q, kv=None) -> bool
"""Whether the kernels take these operands on this GPU (shapes, dtypes, SM90)."""
```

## Configuration

- `FLASHTOPK_PRUNING`: `adaptive` (the default), `always` or `off`. `pruning=` sets it per selector.
- `FLASHTOPK_PRUNE_COSTS="p,c,b"`: overrides this GPU's pruning cost model (see [Hardware and model support](hardware.md)).

## Training and context parallelism

Selection carries no gradient. In training, FlashTopK replaces the forward pass's score table and top-k. The indexer's loss can rescore the returned `topk` keys under autograd.

Under context parallelism, each rank selects for its own queries against the gathered keys. It gets the bits one GPU would, and needs no selection from another rank: each row's threshold comes from a strided sample of its own window. For Megatron-LM's two-chunk layout:

```python
# Rank r of cp holds query chunks r and 2 * cp - 1 - r of a sequence of L keys.
# kv_gathered holds all L keys; q holds this rank's 2 * half queries.
half = q.shape[0] // 2
t = torch.arange(1, half + 1, device="cuda", dtype=torch.int32)
ks = torch.zeros(2 * half, device="cuda", dtype=torch.int32)    # every query sees the sequence from key 0
ke = torch.cat([r * half + t, L - (r + 1) * half + t])          # the causal end of each query
indices = flashtopk.select(q, kv_gathered, weights, ks, ke, kv_scales, topk=2048)
```
