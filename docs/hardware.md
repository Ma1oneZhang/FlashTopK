# Hardware and model support

FlashTopK runs on NVIDIA Hopper (SM90) GPUs, with FP8 or BF16 operands. The kernels target SM90; adaptive pruning's cost model (`PRUNE_COSTS` in [`flashtopk/plan.py`](../flashtopk/plan.py)) follows the GPU itself.

| GPU | architecture | FP8 | BF16 |
|---|---|:---:|:---:|
| H800 | SM90 (9.0) | ✅ | ✅ |
| H20 | SM90 (9.0) | ✅ | ✅ |
| H100 | SM90 (9.0) | ✅ | ✅ |
| H200 | SM90 (9.0) | ✅ | ✅ |
| other SM90 | SM90 (9.0) | ↩️ H800 | ↩️ H800 |

* ✅ **Tuned:** tested, with its own `PRUNE_COSTS` entry.
* ↩️ **Falls back:** uses the `PRUNE_COSTS` of the GPU after the arrow.

## Models

FlashTopK takes indexers with 32 heads of dimension 128 and a ReLU per head; `topk` can be up to 4096.

| model | indexer heads × head dim | supported |
|---|---|:---:|
| [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) | 32 × 128 | ✅ |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | 32 × 128 | ✅ |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 32 × 128 | ✅ |
| [Hy4-preview](https://huggingface.co/tencent/Hy4-preview) | 32 × 128 | ✅ |
| [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2), [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) | 64 × 128 | ❌ |

## How a GPU is matched

1. A word of `torch.cuda.get_device_name()` names a `PRUNE_COSTS` entry (or a `DEVICE_ALIASES` alias of one): that entry, for the call's operand type.
2. Otherwise the SM90 default: H800's entry.
3. `FLASHTOPK_PRUNE_COSTS="p,c,b"` overrides both, for every call.

`FLASHTOPK_PRUNING=off` (or `Selector(pruning="off")`) skips the decision and always scans, which makes the cost model irrelevant.
