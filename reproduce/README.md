# Reproduce FlashTopK's performance

These scripts rerun the comparison of [docs/performance.md](../docs/performance.md) on an SM90 GPU. Each call is one full causal prefill of 128K to 1M keys, with `topk = 2048` in FP8 and BF16. [`pins.toml`](pins.toml) pins every version, down to the commit of each baseline.

## Quick start

```bash
cd reproduce
bash setup.sh                          # pinned environment in .venv, baselines built in third_party/
.venv/bin/python get_inputs.py random  # random inputs; `get_inputs.py hf` downloads the model inputs
GPU=0 bash run.sh                      # every method on every input, into results/
.venv/bin/python report.py             # speedups, prefill times, exactness and figures
```

For a short check, run `get_inputs.py random --lengths 128k`, then `SETS=random LENGTHS=128k bash run.sh`.

**Requirements:** an SM90 GPU with 80 GB or more (H800, H20, H100 or H200), CUDA 13 with `nvcc`, GCC 13+, CMake 3.26+ and Python 3.11 to 3.14 with its headers. Building DeepSelect takes about half an hour with 32 cores.

## Inputs

| set | indexer | source |
|---|---|---|
| `glm` | trained: GLM-5.1 at layers 24 and 48, prefilling 1M tokens of an InfiniteBench book | `get_inputs.py hf` |
| `init` | untrained: the same layers with Megatron-LM initial weights | `get_inputs.py hf` |
| `random` | random q, k and head weights | `get_inputs.py random` |

`get_inputs.py hf` downloads the 1M prefills (36 GB) from [`YangshenDeng/flashtopk-reproduce-inputs`](https://huggingface.co/datasets/YangshenDeng/flashtopk-reproduce-inputs). It cuts the shorter lengths locally. All inputs take about 190 GB on disk; `--quick` downloads layer 24 only (18 GB). [`dump_sglang.py`](dump_sglang.py) regenerates the model sets with SGLang.

## Methods

`flashtopk` fuses scoring with top-k. The three baselines first score an FP32 table, with DeepGEMM `fp8_fp4_mqa_logits` in FP8 or cuDNN in BF16. They differ only in the top-k:

| method | top-k |
|---|---|
| `dense` | FlashInfer `top_k_ragged_transform` |
| `hpc_topk` | HPC-Ops `hpc.topk` |
| `deepselect` | DeepSelect `deep_select.topk` |

Each time is the median of 3 prefills after 1 warmup, measured with CUDA events. `bench.py` checks about 10K rows of each prefill against the exact top-k of the dense scores.

## Results

`report.py` compares every method with `dense`. It draws `results/performance_<set>_<gpu>.png`; `figures.py --out ../assets` redraws the repository's figures from your results. To compare two runs, pass `--reference SAVED_RESULTS` to `report.py`. It fails if FlashTopK's speedup differs by more than 10% or its outputs differ.

Reruns skip finished calls. `run.sh` and `report.py` exit nonzero on a failed call or a wrong row. Each script's header lists its options.
