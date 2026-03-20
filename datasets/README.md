# Datasets for Natural Hallucinations Research

This directory contains datasets for studying natural hallucinations in large language models.
Data files are NOT committed to git due to size. Follow the download instructions below.

## Summary

| Dataset | Source | Size | Disk | Purpose |
|---------|--------|------|------|---------|
| TruthfulQA (generation) | `truthfulqa/truthful_qa` | 817 questions | ~300K | Benchmark for imitative falsehoods |
| TruthfulQA (multiple_choice) | `truthfulqa/truthful_qa` | 817 questions | ~300K | MC-format truthfulness evaluation |
| TriviaQA | `trivia_qa` (rc.nocontext) | 17,944 validation | ~8MB | Closed-book factual QA |
| Natural Questions Open | `google-research-datasets/nq_open` | 87,925 train + 3,610 val | ~5MB | Open-domain QA |
| HaluEval | `pminervini/HaluEval` | 7 configs, ~10K each | ~69MB | Hallucination detection evaluation |

## Dataset Details

### TruthfulQA
- **Path**: `datasets/truthfulqa/generation/` and `datasets/truthfulqa/multiple_choice/`
- **Configs**: `generation` (free-form answers) and `multiple_choice` (MC1/MC2 targets)
- **Features (generation)**: type, category, question, best_answer, correct_answers, incorrect_answers, source
- **Features (MC)**: question, mc1_targets, mc2_targets
- **Relevance**: Core benchmark for "natural hallucinations" -- questions designed to cause models to generate false answers that mimic human misconceptions

### TriviaQA (No Context)
- **Path**: `datasets/triviaqa/validation/`
- **Config**: `rc.nocontext`, validation split only
- **Size**: 17,944 examples
- **Features**: question, question_id, question_source, entity_pages, search_results, answer
- **Relevance**: Factual QA benchmark for evaluating hallucination in closed-book settings

### Natural Questions Open
- **Path**: `datasets/natural_questions/`
- **Splits**: train (87,925) and validation (3,610)
- **Features**: question, answer (list of acceptable answers)
- **Relevance**: Open-domain QA benchmark for factual knowledge in LLMs

### HaluEval
- **Path**: `datasets/halueval/{config}/`
- **Configs**: dialogue, dialogue_samples, general, qa, qa_samples, summarization, summarization_samples
- **Sizes**: 10,000 examples each (general: 4,507)
- **Relevance**: Provides labeled hallucinated vs. correct responses across QA, dialogue, and summarization

## Download Script

To re-download all datasets from scratch:

```python
from datasets import load_dataset
import os

# TruthfulQA
for config in ["generation", "multiple_choice"]:
    ds = load_dataset("truthfulqa/truthful_qa", config)
    ds.save_to_disk(f"datasets/truthfulqa/{config}")

# TriviaQA (validation only, no context)
ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
ds.save_to_disk("datasets/triviaqa/validation")

# Natural Questions Open
ds = load_dataset("google-research-datasets/nq_open")
ds.save_to_disk("datasets/natural_questions/")

# HaluEval (all configs)
for config in ["dialogue", "dialogue_samples", "general", "qa", "qa_samples",
               "summarization", "summarization_samples"]:
    ds = load_dataset("pminervini/HaluEval", config)
    ds.save_to_disk(f"datasets/halueval/{config}")
```

## Loading Datasets

```python
from datasets import load_from_disk

# Example: load TruthfulQA generation
truthfulqa = load_from_disk("datasets/truthfulqa/generation")

# Example: load HaluEval QA samples
halueval_qa = load_from_disk("datasets/halueval/qa_samples")
```

## Sample Files

Each dataset directory includes a `*_sample.json` file with 3-5 example entries for quick inspection without loading the full dataset.

## Notes for Experiments

### Primary Datasets for "Natural Hallucinations" Research
1. **TruthfulQA** - Most relevant for studying imitative falsehoods that transfer across models
2. **HaluEval** - For training/evaluating hallucination detection methods
3. **TriviaQA / NQ Open** - For factual QA hallucination evaluation

### Key Properties to Investigate
- Which questions in TruthfulQA cause consistent hallucinations across multiple models?
- Do hallucinations on certain questions persist after fine-tuning?
- Can models recognize their own hallucinations when presented in isolation (snowballing)?
