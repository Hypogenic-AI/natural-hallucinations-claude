# Natural Hallucinations in Large Language Models

Investigating whether certain LLM hallucinations are "natural" — robust to rephrasing, resistant to self-detection, and transferable across models. Tested on the full TruthfulQA benchmark (817 questions) across 4 OpenAI models.

## Key Findings

- **27 universal hallucinations** (3.3% of questions) fool all 4 models tested
- **76% persistence**: hallucinations survive question rephrasing (5 variants)
- **10-16% self-detection**: models actively prefer their own wrong answer over the correct one in an A/B test (vs 66-95% on control questions)
- **Significant transfer**: pairwise Jaccard similarity 0.19-0.41, all 3-6× above random (p<0.001)
- **Temporal prediction**: GPT-3.5 errors predict GPT-4.1 errors (r=0.256, p<0.0001)

## Project Structure

```
├── REPORT.md              # Full research report with results and analysis
├── planning.md            # Research plan and motivation
├── literature_review.md   # Literature survey
├── resources.md           # Resource catalog
├── src/
│   ├── run_exp1_parallel.py   # Exp 1: Cross-model hallucination survey
│   ├── run_exp234.py          # Exps 2-4: Robustness, self-detection, transfer
│   ├── analysis.py            # Statistical analysis and visualization
│   └── experiment.py          # Combined experiment script (reference)
├── results/
│   ├── raw/                   # Raw JSON results per experiment
│   ├── cache/                 # Cached API responses
│   ├── plots/                 # Generated visualizations
│   └── analysis.json          # Compiled analysis results
├── datasets/                  # TruthfulQA, TriviaQA, NQ, HaluEval
├── papers/                    # Downloaded research papers
└── code/                      # Cloned baseline repositories
```

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy pandas matplotlib seaborn scipy datasets tqdm

# Run experiments (requires OPENAI_API_KEY)
python src/run_exp1_parallel.py    # ~10 min, ~6500 API calls
python src/run_exp234.py           # ~15 min, ~8500 API calls
python src/analysis.py             # Analysis + plots
```

## Models Tested

| Model | Hallucination Rate | Year |
|-------|-------------------|------|
| GPT-4.1 | 8.9% | 2025 |
| GPT-4o | 12.5% | 2024 |
| GPT-3.5-turbo | 21.3% | 2023 |
| GPT-4o-mini | 24.9% | 2024 |

See [REPORT.md](REPORT.md) for the full analysis.
