![FlashTopK, Tencent Hy](assets/banner.png)

# FlashTopK: Fast Fused Top-k Selection for Sparse Attention Training and Inference

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue"></a>
  <a href="https://arxiv.org/abs/2610.32057"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2610.32057-b31b1b"></a>
</p>

FlashTopK is a fast, exact and deterministic fused top-k selection kernel for learned sparse attention, in both training and inference.

On a full prefill, FlashTopK is 1.7-3.3x faster than dense scoring + FlashInfer top-k (DeepGEMM / cuDNN scoring) on trained indexers, and 1.2-2.0x faster on untrained indexers.

## News

[2026/10] We have released **[HyScale: Scaling Long-Context Training for Learned Sparse Attention](https://arxiv.org/abs/2610.32057)**.

## Highlights

- **Fast:** the fastest open-source implementation.
- **Robust:** stays fast on trained and untrained indexers. Adaptive key pruning decides per call whether to skip keys by a score bound.
- **Exact:** the true top-k. Rows the sampled threshold misses are rescored on the GPU.
- **Deterministic:** the same bits on every run. Ties at the k-th score keep the smallest key indices.
- **CP-compatible:** each row's threshold comes from a strided sample of its own window, so [context parallelism](docs/api.md#training-and-context-parallelism) needs no selection from another rank.

## Support

| Sparse attention | Models | GPUs | Operands |
|---|---|---|---|
| Token-level selection, as in [DeepSeek Sparse Attention (DSA)](https://arxiv.org/abs/2512.02556) | [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1), [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2), [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), [Hy4-preview](https://huggingface.co/tencent/Hy4-preview): indexers of 32 heads × 128 | NVIDIA Hopper (SM90): H800, H20, H100, H200; NVIDIA SM100 and SM103 | FP8 or BF16, `topk` up to 4096 |

[Hardware and model support](docs/hardware.md) lists which GPUs are tuned.

## Getting started

### Install

FlashTopK needs Python 3.10+, PyTorch 2.6+, Triton 3.3+ and CuTe DSL (`nvidia-cutlass-dsl`) 4.5+. SM100 GPUs also need a CUDA 12.8+ driver.

```bash
pip install -e .            # torch >= 2.6, triton >= 3.3, nvidia-cutlass-dsl >= 4.5, apache-tvm-ffi
pip install -e .[test]      # pytest for the test suite
```

### Quick start

For each query `t`, FlashTopK selects the `topk` keys `s` of its window `[ks[t], ke[t])` with the largest indexer score:

```math
I_{t,s} = \sigma_s \sum_{h=1}^{H_I} w_{t,h}\,\mathrm{ReLU}\big(\mathbf{q}_{t,h}^{\top}\mathbf{k}_s\big)
```

Here **q**<sub>t,h</sub> is head h of `q[t]`, **k**<sub>s</sub> is `kv[s]`, w<sub>t,h</sub> is head h's weight in `weights[t]`, σ<sub>s</sub> is `kv_scales[s]` (1 for BF16 keys) and H<sub>I</sub> = 32 is the number of heads. A causal prefill chunk of 4096 queries over 128K FP8 keys:

```python
import torch
import flashtopk

num_q, seq_len_kv = 4096, 128 * 1024
q = torch.randn(num_q, 32, 128, device="cuda").to(torch.float8_e4m3fn)        # 32 heads of dimension 128
kv = torch.randn(seq_len_kv, 128, device="cuda").to(torch.float8_e4m3fn)       # one key per token
kv_scales = torch.rand(seq_len_kv, device="cuda") * 0.02 + 0.01                # FP8 keys need per-key scales
weights = torch.randn(num_q, 32, device="cuda") * 0.1                          # per-head weights, float32

# The queries are the last num_q tokens: query i sees keys [0, seq_len_kv - num_q + i + 1).
ke = torch.arange(seq_len_kv - num_q + 1, seq_len_kv + 1, device="cuda", dtype=torch.int32)
ks = torch.zeros_like(ke)

indices = flashtopk.select(q, kv, weights, ks, ke, kv_scales, topk=2048)  # [4096, 2048] int32
```

The first call in a process compiles the kernels it needs, about 15 seconds in all. `Selector.plan()` compiles them up front.

### Usage

```python
flashtopk.select(q, kv, weights, ks, ke, kv_scales=None, topk=2048, ...)  # indices; return_scores=True adds scores
selector = flashtopk.Selector(topk=2048, dtype=...).plan(max_rows, max_kv)  # owns the workspace, compiles up front
selector.run(q, kv, weights, ks, ke, kv_scales)                             # select's arguments and outputs
torch.ops.flashtopk.select(...)                                             # select as a custom op, for torch.compile
flashtopk.windows_from_cu_seqlens(cu_seqlens_q, cu_seqlens_k)               # causal windows of packed sequences
```

The [API reference](docs/api.md) covers every argument, the workspace, `last_info`, configuration and training with context parallelism.

## Performance

Speedup of one full prefill (query rows = keys = the sequence length) over dense scoring + FlashInfer top-k on H800, with FP8 or BF16 operands. The [performance report](docs/performance.md) has H20 and SM100 as well.

### Trained indexer

<p align="center"><img src="assets/performance_trained_h800.png" alt="FlashTopK against public top-k selectors on a trained indexer, H800" width="100%"></p>

FlashTopK is the fastest at every length, and its lead grows with the length.

### Untrained indexer

<p align="center"><img src="assets/performance_untrained_h800.png" alt="FlashTopK against public top-k selectors on an untrained indexer, H800" width="100%"></p>

The score bound skips few keys, so FlashTopK mostly scans; it still leads at every length.

The [performance report](docs/performance.md) describes the setup and the methods compared. It also has the results on random indexer weights and keys. [`reproduce/`](reproduce/README.md) provides pinned setup, inputs, comparison benchmarks and scripts to regenerate the figures.

## Roadmap

- Block-level learned sparse attention, such as CSA in [DeepSeek-V4](https://arxiv.org/abs/2606.19348) and QSA in [Qwen3.8-Flash-Next](https://arxiv.org/abs/2608.30320).
- Paged KV cache; today the keys must be contiguous, with one contiguous window per query row.
- Indexers with other head counts, such as the 64 heads of DeepSeek-V3.2.

## License

FlashTopK is released under the Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details.

## Acknowledgments

FlashTopK builds on [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (the SM100 scan pipeline, which the pruning and refine kernels share), [LiteTopK](https://github.com/Heisenberg-Yin/LiteTopK) (gate calibration and candidate selection) and [DoSA](https://arxiv.org/abs/2610.13990) (the score bound of key pruning). [`NOTICE`](NOTICE) lists the files each part derives from.

## Citation

```bibtex
@misc{deng2026hyscale,
  title         = {HyScale: Scaling Long-Context Training for Learned Sparse Attention},
  author        = {Yangshen Deng and Xinjing Huang and Yulong Wang and Yunzhuo Liu and Jun Hu and Shuyong Tan and Chun-Chih Tseng and Mingji Han and Ray Ying and Key Zhang and Luo Mai},
  year          = {2026},
  eprint        = {2610.32057},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.32057},
}
```
