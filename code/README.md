# Code Repositories for Natural Hallucinations Research

## Cloned Repositories

### 1. TruthfulQA
- **URL**: https://github.com/sylinrl/TruthfulQA
- **Location**: `code/TruthfulQA/`
- **Purpose**: Official implementation of TruthfulQA benchmark for measuring model truthfulness
- **Key Files**:
  - `truthfulqa/evaluate.py` - Evaluation code
  - `truthfulqa/utilities.py` - Helper functions
  - `data/` - Question data and reference answers
- **How to Use**:
  ```bash
  cd code/TruthfulQA
  pip install -e .
  python -m truthfulqa.evaluate --model_name <model> --model_path <path>
  ```

### 2. SelfCheckGPT
- **URL**: https://github.com/potsawee/selfcheckgpt
- **Location**: `code/selfcheckgpt/`
- **Purpose**: Zero-resource black-box hallucination detection via self-consistency
- **Key Files**:
  - `selfcheckgpt/` - Core implementation
  - `demo/` - Example notebooks
- **How to Use**:
  ```python
  from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore, SelfCheckNgram
  selfcheck = SelfCheckBERTScore()
  scores = selfcheck.predict(sentences, sampled_passages)
  ```

### 3. INSIDE / EigenScore
- **URL**: https://github.com/alibaba/eigenscore
- **Location**: `code/INSIDE_eigenscore/`
- **Purpose**: Hallucination detection using LLM internal states and eigenvalue-based consistency
- **Key Files**:
  - `eigenscore.py` - Core EigenScore implementation
  - `feature_clip.py` - Feature clipping for overconfident hallucinations
- **Key Insight**: Uses eigenvalues of sentence embedding covariance matrix to detect semantic divergence

### 4. Snowballed Hallucination
- **URL**: https://github.com/Nanami18/Snowballed_Hallucination
- **Location**: `code/snowball_hallucination/`
- **Purpose**: Study of hallucination snowballing - how early mistakes lead to more errors
- **Key Files**:
  - Code for primality testing, senator search, and graph connectivity experiments
- **Key Insight**: LLMs can identify 67-87% of their own mistakes when presented in isolation

### 5. HaluEval
- **URL**: https://github.com/RUCAIBox/HaluEval
- **Location**: `code/HaluEval/`
- **Purpose**: Hallucination evaluation benchmark with 35K generated samples
- **Key Files**:
  - `evaluation/` - Evaluation scripts
  - `data/` - Benchmark data
- **How to Use**:
  ```bash
  cd code/HaluEval/evaluation
  python evaluation.py
  ```

## Relevance to Natural Hallucinations Research

These repositories provide:

1. **Benchmarks**: TruthfulQA for imitative falsehoods, HaluEval for diverse hallucinations
2. **Detection Methods**: SelfCheckGPT (self-consistency), INSIDE/EigenScore (internal states)
3. **Analysis Tools**: Snowball paper's verification methodology for studying if models recognize their errors

## Key Questions to Investigate

1. **Transferability**: Do hallucinations on TruthfulQA questions transfer across different model families?
2. **Recognizability**: Can models identify hallucinated statements when presented in isolation (snowballing)?
3. **Persistence**: Do certain hallucinations persist after fine-tuning or RLHF?
4. **Detection**: Which detection methods (consistency vs internal states) work better for "natural" hallucinations?

## Installation Notes

Most repositories require:
```bash
pip install transformers torch datasets
```

For INSIDE/EigenScore, also need:
```bash
pip install scipy numpy
```
