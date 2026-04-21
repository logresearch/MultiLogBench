# MultiLogBench

MultiLogBench is a multilingual benchmark for automated logging statement generation. Each sample provides a callable with one developer-written logging statement removed (`input_initial`), the original callable (`function_content`), and a structured target logging statement (`target_log`). The benchmark covers C/C++, C#, Go, Java, JavaScript, and Python.

## Project Tree

```text
MultiLogBench/
|-- data/
|   |-- repository_snapshot/
|   |   |-- samples.jsonl
|   |   |-- manifest.json
|   |   |-- stats.json
|   |   |-- postprocess_report.json
|   |   `-- splits/
|   |       |-- train.jsonl
|   |       |-- valid.jsonl
|   |       |-- test.jsonl
|   |       |-- manifest.json
|   |       `-- stats.json
|   |-- revision_history/
|   |   |-- accepted_pool.jsonl
|   |   |-- manifest.json
|   |   |-- stats.json
|   |   `-- postprocess_report.json
|   `-- revision_history_transformed/
|       |-- accepted_pool.jsonl
|       |-- manifest.json
|       |-- stats.json
|       `-- transform_report.json
|-- multilogbench/
|   `-- logging_dataset/
|       |-- config.py
|       |-- core.py
|       `-- runtime.py
|-- scripts/
|   |-- run_llms.py
|   `-- evaluate_logging_metrics.py
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Data

| Dataset | Path | Rows | Use |
|---|---:|---:|---|
| Repository-snapshot full set | `data/repository_snapshot/samples.jsonl` | 75,292 | Full cleaned snapshot dataset |
| Repository-snapshot train | `data/repository_snapshot/splits/train.jsonl` | 51,120 | Training-oriented split |
| Repository-snapshot valid | `data/repository_snapshot/splits/valid.jsonl` | 6,426 | Retrieval examples and validation |
| Repository-snapshot test | `data/repository_snapshot/splits/test.jsonl` | 6,419 | In-distribution test split |
| Revision-history test | `data/revision_history/accepted_pool.jsonl` | 744 | Main historical benchmark |
| Transformed revision-history test | `data/revision_history_transformed/accepted_pool.jsonl` | 744 | Leakage-resistant transformed benchmark |

The core JSONL fields are:

| Field | Meaning |
|---|---|
| `index` | Stable sample identifier |
| `language` | Programming language |
| `repo_name` | Source repository |
| `function_name` | Callable name |
| `function_content` | Original callable with the target logging statement |
| `input_initial` | Callable after removing the target logging statement |
| `target_log` | Gold logging statement fields: `line`, `level`, `message`, `vars`, `statement` |
| `target_log_normalized` | Normalized logging level, message template, and variable fields |

Revision-history samples additionally include parent/child commit metadata and `parent_function_content`.

Large JSONL files are tracked with Git LFS through `.gitattributes`. After cloning, run:

```bash
git lfs install
git lfs pull
```

## Installation

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The evaluator uses `tree-sitter-languages` to parse generated code snippets and extract logging statements.

## Running Model Generation

The public runner writes a run directory containing:

| File | Meaning |
|---|---|
| `selected_gold.jsonl` | Gold subset used in the run |
| `generations.jsonl` | Raw generated revised code keyed by sample `index` |
| `run_config.json` | Run configuration |

Dry-run smoke test:

```bash
python scripts/run_llms.py \
  --dry-run \
  --max-samples 2 \
  --run-name smoke
```

OpenAI-compatible API:

```bash
export OPENAI_API_KEY=your_api_key

python scripts/run_llms.py \
  --provider openai-compatible \
  --model gpt-5.4 \
  --base-url https://api.openai.com/v1 \
  --dataset data/revision_history/accepted_pool.jsonl \
  --strategy rag \
  --max-workers 8
```

Gemini API:

```bash
export GEMINI_API_KEY=your_api_key

python scripts/run_llms.py \
  --provider google-genai \
  --model gemini-2.5-flash \
  --dataset data/revision_history/accepted_pool.jsonl \
  --strategy rag \
  --max-workers 8
```

Useful filters:

```bash
python scripts/run_llms.py --dry-run --languages java,python --max-samples 10
python scripts/run_llms.py --dry-run --sample-indexes 1,2,3
```

## Evaluation

Evaluate a run directory:

```bash
python scripts/evaluate_logging_metrics.py \
  --run-dir runs/dry-run/smoke
```

This produces, by default:

| File | Meaning |
|---|---|
| `metrics.json` | Aggregate metrics |
| `metrics_details.jsonl` | Per-sample metric details |
| `predictions.jsonl` | Normalized predicted logging statement records |

Evaluate raw generations explicitly:

```bash
python scripts/evaluate_logging_metrics.py \
  --gold data/revision_history/accepted_pool.jsonl \
  --pred runs/MODEL_NAME/historical_test/generations.jsonl \
  --pred-format generations \
  --pa-mode target-line-change \
  --output metrics.json \
  --details-out metrics_details.jsonl \
  --predictions-out predictions.jsonl
```

Evaluate structured predictions:

```bash
python scripts/evaluate_logging_metrics.py \
  --gold data/repository_snapshot/splits/test.jsonl \
  --pred predictions.jsonl \
  --pred-format structured \
  --pa-mode exact-line \
  --output metrics.json
```

`--pa-mode auto` uses `target-line-change` for raw `generations.jsonl` and `exact-line` for structured `predictions.jsonl`.

## Prediction Format

Structured prediction files should use one JSON object per line:

```json
{"index":123,"pred_log":{"line":7,"level":"warn","message":"\"retry failed {}\", e.getMessage()","vars":["e.getMessage()"],"statement":"logger.warn(\"retry failed {}\", e.getMessage());"}}
```

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `index` | integer | Must match a gold sample index |
| `pred_log.line` | integer or null | 1-based callable-relative logging statement line |
| `pred_log.level` | string or null | Predicted logging level |
| `pred_log.message` | string | Predicted logging payload text |
| `pred_log.vars` | string array | Predicted runtime variables or expressions |
| `pred_log.statement` | string | Full predicted logging statement |

If prediction fails, still emit a record with null/empty `pred_log` fields and an optional `status` or `error` field.

## Metrics

The evaluator reports:

| Metric | Meaning |
|---|---|
| PA | Position Accuracy |
| FA | Framework Accuracy |
| LA | Logging Level Accuracy |
| AOD | Average Ordinal Distance for logging level |
| BLEU-4 | N-gram overlap for logging text |
| ROUGE-L | Longest-common-subsequence similarity for logging text |
| PMR | Precisely Matched Rate for logging variables |
| Precision / Recall / F1 | Set-based logging variable metrics |
| CCS | Weakly cascaded composite score: `PA * (0.5 + 0.25 * FA + 0.25 * ((LA + ROUGE-L + F1) / 3))` |

By default, PA is averaged over all samples, while FA, LA, AOD, BLEU-4, ROUGE-L, PMR, Precision, Recall, and F1 are conditioned on samples with PA = 1. CCS is computed from the aggregate PA, FA, LA, ROUGE-L, and F1 values.

## Citation

@misc{zhong2026singlelanguageevidenceinsufficientautomated,
      title={Single-Language Evidence Is Insufficient for Automated Logging: A Multilingual Benchmark and Empirical Study with LLMs}, 
      author={Renyi Zhong and Yichen Li and Yulun Wu and Jinxi Kuang and Yintong Huo and Michael R. Lyu},
      year={2026},
      eprint={2604.17529},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2604.17529}, 
}

## License

The code and documentation in this repository are released under the MIT License. Dataset samples are derived from open-source repositories.
