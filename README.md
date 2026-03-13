# Material Bench

[日本語版 README はこちら](./README.ja.md)

Inference and evaluation scripts for material science benchmarks (MaterialFigBench, MaterialBENCH).

## Overview

| Dataset | Format | # Questions | Data Source |
|---|---|---|---|
| **MaterialFigBench** | Image + Text (multimodal) | 137 | [HuggingFace](https://huggingface.co/datasets/omron-sinicx/MaterialFigBench) / [Paper](https://arxiv.org/abs/2409.03161) |
| **MaterialBENCH choice** | Multiple choice (4 options) | 164 | [HuggingFace](https://huggingface.co/datasets/omron-sinicx/MaterialBENCH) / [Paper](https://arxiv.org/abs/2603.11414) |
| **MaterialBENCH free** | Free-form answer | 144 | Same as above |

> Note: This is an unofficial implementation and is not intended to exactly reproduce the results reported in the papers.

Inference and evaluation are performed using an OpenAI-compatible API (e.g. vLLM).

## Installation

```bash
pixi install
```

## Usage

```bash
# Run a single dataset
pixi run python scripts/run_benchmark.py \
  --dataset materialbench-choice \
  --api-base http://localhost:8001

# Run all datasets
pixi run python scripts/run_benchmark.py \
  --dataset all \
  --api-base http://localhost:8001

# Run with LLM judge
pixi run python scripts/run_benchmark.py \
  --dataset materialbench-free \
  --api-base http://localhost:8001 \
  --use-llm-judge

# Apply LLM judge only to an existing CSV result
pixi run python scripts/run_benchmark.py \
  --llm-judge-only \
  --dataset materialbench-free \
  --api-base http://localhost:8001
```

### Main Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Target dataset (material-figbench / materialbench-choice / materialbench-free / all) | Required |
| `--api-base` | OpenAI-compatible API base URL | `http://localhost:8001` |
| `--model` | Model name | `Qwen/Qwen3.5-397B-A17B-FP8` |
| `--output-dir` | Directory to save results | `./results` |
| `--num-threads` | Number of parallel threads | `1` |
| `--max-samples` | Maximum number of samples (0 = all) | `0` |
| `--use-llm-judge` | Run LLM evaluation after inference | `False` |
| `--llm-judge-only` | Apply LLM evaluation only to an existing CSV | `False` |
| `--judge-csv` | CSV file to apply LLM evaluation to | `None` |
| `--judge-api-base` | API base URL for LLM judge | Same as `--api-base` |
| `--judge-model` | Model name for LLM judge | Same as `--model` |
| `--judge-num-threads` | Number of parallel threads for LLM judge | Same as `--num-threads` |

### Output Files

```
results/
├── {dataset}_YYYYMMDD_HHMMSS.csv   # Inference results
├── summary_YYYYMMDD_HHMMSS.json    # Evaluation summary
└── *_judged.csv                    # LLM judge results
```

The CSV contains columns such as `question_id`, `prediction`, and `correct`. When `--use-llm-judge` is used, `llm_judge_correct` and `llm_judge_reason` are also added.

## Benchmark Results

> **Note:** 6 questions were excluded from MaterialFigBench due to missing image files in the data source; evaluation is currently performed on 131/137 questions. Re-evaluation is planned after the source data is fixed.

![Benchmark Results](./asset/benchmark_results.png)

- Pull requests to add more results are welcome.

### Accuracy (Exact Match)

| Model | FigBench | Choice | Free | Overall |
|-------|----------|--------|------|---------|
| Qwen3.5-397B-A17B | 22.14% (29/131) | 77.44% (127/164) | 52.08% (75/144) | 52.62% (231/439) |
| Qwen3.5-27B | 22.90% (30/131) | 67.68% (111/164) | 49.31% (71/144) | 48.29% (212/439) |

### Accuracy with LLM Judge (Semantic Match)

- `--judge-model = "Qwen/Qwen3.5-397B-A17B-FP8"`

| Model | FigBench | Choice | Free | Overall |
|-------|----------|--------|------|---------|
| Qwen3.5-397B-A17B | 41.22% (54/131) | 77.44% (127/164) | 86.11% (124/144) | 69.48% (305/439) |
| Qwen3.5-27B | 42.75% (56/131) | 67.68% (111/164) | 84.03% (121/144) | 65.60% (288/439) |

## License

MIT
