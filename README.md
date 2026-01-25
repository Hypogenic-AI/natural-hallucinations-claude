# Natural Hallucinations in Large Language Models

This research project investigates "natural hallucinations" - factual errors that multiple LLMs consistently produce, are robust to rephrasing, and difficult for models to recognize as mistakes.

## Key Findings

- **7% of TruthfulQA questions cause consistent failures across 3+ models** (GPT-4o, Claude 3.5 Sonnet, GPT-3.5-turbo, Llama 3 70B)
- **57% robustness to paraphrasing** - these errors persist even when questions are reworded
- **Only 24% self-recognition** - models struggle to identify their own errors when asked directly
- **20% error inheritance** - GPT-4o inherited 1 in 5 of GPT-3.5's hallucinations
- **Common patterns**: Stereotype traps ("most people love X"), misquotations (Nixon), legal misconceptions

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run experiments (requires OPENAI_API_KEY and OPENROUTER_API_KEY)
python src/experiment.py

# Generate visualizations
python src/visualizations.py
```

## Results at a Glance

| Model | Accuracy on TruthfulQA |
|-------|------------------------|
| GPT-4o | 94% |
| Claude 3.5 Sonnet | 87% |
| Llama 3 70B | 81% |
| GPT-3.5-turbo | 75% |

## Project Structure

```
natural-hallucinations-claude/
├── src/
│   ├── experiment.py         # Main experiment code
│   └── visualizations.py     # Plotting and analysis
├── results/
│   ├── exp1_results.json     # Cross-model transfer results
│   ├── exp2_results.json     # Paraphrasing robustness results
│   ├── exp3_results.json     # Self-recognition results
│   ├── exp4_results.json     # Temporal analysis results
│   ├── all_results_summary.json
│   └── plots/                # Generated visualizations
├── datasets/
│   └── truthfulqa/           # TruthfulQA dataset
├── papers/                   # Related research papers
├── code/                     # Reference implementations
├── planning.md               # Research plan
├── REPORT.md                 # Full research report
└── README.md                 # This file
```

## Methodology

1. **Experiment 1**: Test 100 TruthfulQA questions across 4 models, identify questions where 3+ models fail
2. **Experiment 2**: Generate paraphrases of natural hallucinations, test if errors persist
3. **Experiment 3**: Ask models if their own wrong answers are correct
4. **Experiment 4**: Compare error patterns between GPT-3.5 and GPT-4o

## Requirements

- Python 3.10+
- OpenAI API key (for GPT-4o, GPT-3.5-turbo)
- OpenRouter API key (for Claude, Llama)

## Installation

```bash
# Create environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install openai anthropic datasets scipy numpy pandas matplotlib seaborn tqdm requests
```

## Citation

If you use this work, please cite:

```
@misc{natural-hallucinations-2026,
  title={Natural Hallucinations: Cross-Model Error Transfer in Large Language Models},
  year={2026},
  howpublished={Research project},
}
```

## See Also

- [REPORT.md](REPORT.md) - Full research report with detailed findings
- [planning.md](planning.md) - Research plan and methodology
- [literature_review.md](literature_review.md) - Background on hallucination research
