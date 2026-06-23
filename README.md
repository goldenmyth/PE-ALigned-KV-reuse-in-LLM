# RoPE-Aware KV-Cache Reuse for Multi-Document RAG

This repository contains an experimental implementation of **RoPE-aware KV-cache reuse** for multi-document Retrieval-Augmented Generation scenarios.

The project explores whether independently precomputed document KV-caches can be reused inside a longer prompt when the model uses **Rotary Positional Embeddings (RoPE)**. The main idea is to correct the positional mismatch by applying an additional RoPE phase shift to cached key tensors before assembling them into a shared Hugging Face `DynamicCache`.

## Report

The full technical report is available here:

[Report.pdf](Report.pdf)

## Motivation

In RAG systems, the same documents may appear in many different prompts. Recomputing their KV-cache every time is expensive, especially when prompts contain long or multiple documents.

Standard KV-cache reuse works well for fixed prefixes, but document reuse is harder. A document can be cached independently from position `0`, then later inserted after a system prompt, other documents, or a question. For RoPE-based models, this is not positionally equivalent: cached keys depend on token positions.

This repository implements and evaluates a simple positional correction strategy for this problem.

## Method

The project compares three inference strategies:

* **Baseline**
  Full prefill over the complete prompt. This is the reference method.

* **Naive**
  Independent precomputation of document KV-caches followed by direct cache concatenation, without positional correction.

* **Aligned / PEA**
  Independent precomputation of document KV-caches, followed by RoPE-aware phase shifting of cached key tensors according to their final offsets in the assembled prompt.

Only cached keys `K` are shifted. Cached values `V` are left unchanged because RoPE affects query-key positional scoring, not value aggregation.

## Main Results

Experiments were run with `Qwen/Qwen2.5-7B-Instruct` on multi-hop QA, summarization, and synthetic retrieval tasks.

On RULER Needle-in-a-Haystack retrieval tasks, positional alignment consistently improved over naive cache reuse:

| Dataset  | Metric |  Naive | Aligned | Relative improvement |
| -------- | -----: | -----: | ------: | -------------------: |
| RULER 4k |     F1 | 0.4297 |  0.5665 |               +31.8% |
| RULER 4k |     EM | 0.3780 |  0.5360 |               +41.8% |
| RULER 8k |     F1 | 0.4737 |  0.6610 |               +39.5% |
| RULER 8k |     EM | 0.4200 |  0.6380 |               +51.9% |

The method was less effective on MuSiQue and SAMSum. This is expected: RoPE alignment fixes positional mismatch, but it does not reconstruct the inter-segment self-attention that would exist in full-prefill inference.

## Key Features

* RoPE-aware phase shifting for cached key tensors.
* Segmented KV-cache precomputation.
* Hugging Face `DynamicCache` assembly.
* Comparison against full-prefill and naive cache concatenation.
* Evaluation on MuSiQue, SAMSum, and RULER.
* Numerical stability patch for fp16 attention overflow during decoding.
* Validation scripts for RoPE shift behavior.

## Repository Structure

```text
.
├── main_benchmark.py              # Main benchmark entry point
├── config.yaml                    # Model, dataset and experiment settings
├── requirements.txt               # Python dependencies
├── README.md                      # Project description
│
├── src/
│   ├── utils_rope.py              # RoPE rotation and cache alignment logic
│   ├── utils_cache.py             # KV-cache extraction and assembly
│   ├── utils_data.py              # Dataset-specific prompt construction
│   ├── utils_metrics.py           # Evaluation metrics
│   ├── attention_patch.py         # Selective upcasting for fp16 stability
│   └── config_loader.py           # Config loading utilities
│
├── scripts/
│   └── run_validation.py          # RoPE validation checks
│
└── Report.pdf                     # Full technical report
```

## Installation

```bash
git clone <repo-url>
cd <repo-name>

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

A CUDA-capable GPU is expected. The experiments were developed for a single NVIDIA V100 32GB environment.

## Configuration

Main settings are stored in `config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"

benchmark:
  save_dir: "results"
  seed: 42

strategies:
  - "Aligned"
  - "Naive"
```

Dataset settings can be enabled or disabled from the same config file.

## Running Benchmarks

```bash
python main_benchmark.py
```

The benchmark runs:

1. Full-prefill Baseline.
2. Independent segment precomputation.
3. Aligned cache assembly.
4. Naive cache assembly.
5. Metric computation and CSV export.

## Running RoPE Validation

```bash
python scripts/run_validation.py
```

The validation script checks:

* reversibility of RoPE shifts;
* attention-score invariance under consistent query/key shifts;
* consistency with model-generated shifted position IDs.

## Numerical Stability

During segmented-cache decoding in fp16, some runs produced `Inf` or `NaN` values in attention scores. This was caused by overflow in `QK^T / softmax` on NVIDIA V100, where bfloat16 is not available.

The repository includes a selective upcasting patch that computes the numerically sensitive attention operation in float32 during decoding with a past cache. This removes NaN generation artifacts without running the full model in float32.

## Limitations

This project does not claim full equivalence with full-prefill inference.

Positional encoding aligned corrects the RoPE positional phase of cached keys, but independently cached document segments still miss inter-segment self-attention. As a result, the method works best for retrieval-heavy tasks, where positional correctness is especially important, and is less reliable for tasks requiring strong cross-document reasoning or global summarization.

Latency and memory benchmarks are not the main focus of the current version. The repository focuses on correctness, output quality, and diagnostic analysis.

## Summary

RoPE-aware positional alignment is a useful correction layer for segmented KV-cache reuse. It improves naive cache concatenation on long-context retrieval tasks, but it does not replace full-prefill inference. A natural next step is to combine positional alignment with selective recomputation or CacheBlend-like cache fusion methods.
