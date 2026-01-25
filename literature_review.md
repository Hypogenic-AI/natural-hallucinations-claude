# Literature Review: Natural Hallucinations in Large Language Models

## Research Hypothesis
Some hallucinations produced by large language models (LLMs) are difficult for the models to recognize, even when all relevant information is present. These "natural hallucinations" may be robust to additional context or question phrasing, and may transfer across models or persist after training.

## Research Area Overview

LLM hallucinations—the generation of fluent but factually incorrect content—represent a fundamental challenge for deploying LLMs in real-world applications. While hallucinations are often attributed to knowledge gaps, emerging research suggests a more nuanced picture: some hallucinations arise from the training objective itself, leading models to reproduce common human misconceptions rather than generate truthful content.

This literature review synthesizes findings on hallucination detection, robustness, transferability, and the mechanisms underlying these failures.

---

## Key Papers

### 1. TruthfulQA: Measuring How Models Mimic Human Falsehoods
**Authors**: Lin, Hilton, Evans (2021)
**Source**: arXiv:2109.07958, 2693 citations
**File**: `papers/2109.07958_TruthfulQA.pdf`

**Key Contributions**:
- Introduces concept of "imitative falsehoods" - false answers with high likelihood on training distribution
- 817 questions across 38 categories designed to elicit these falsehoods
- Demonstrates **inverse scaling**: larger models are *less* truthful (GPT-3-175B: 58% truthful vs human: 94%)

**Critical Findings for Our Hypothesis**:
1. **Transfer across models**: GPT-Neo/J shows similar inverse scaling to GPT-3 without adversarial filtering, suggesting imitative falsehoods transfer between models with similar training distributions
2. **Robustness to paraphrasing**: Truthfulness scores don't change substantially on paraphrased questions
3. **Not syntactic artifacts**: Control questions with same syntax but different content show normal scaling (larger = better)

**Methodology**: Human evaluation of generated answers; automated GPT-judge classifier (90-96% accuracy)

**Relevance**: TruthfulQA provides the foundational benchmark for studying "natural hallucinations" that are difficult to fix by scaling alone.

---

### 2. How Language Model Hallucinations Can Snowball
**Authors**: Zhang, Press, Merrill, Liu, Smith (2023)
**Source**: arXiv:2305.13534
**File**: `papers/2305.13534_hallucination_snowball.pdf`

**Key Contributions**:
- Demonstrates "hallucination snowballing" - models make errors they can separately recognize as wrong
- ChatGPT identifies 67% of its own mistakes; GPT-4 identifies 87%
- Created 3 datasets: primality testing, senator search, graph connectivity

**Critical Findings for Our Hypothesis**:
1. **Self-recognition**: Models "know" their errors are wrong when asked separately
2. **Over-commitment**: LMs commit to early mistakes for consistency, generating supporting false claims
3. **Robustness to interventions**: Snowballed hallucinations persist with temperature changes (0.0, 0.6, 0.9)
4. **Limits of prompting**: Even with "Let's think step by step", 95% of failures still show snowballing

**Mechanism**: Initial committal (Yes/No first token) + inherently sequential problems (beyond single-step transformer reasoning)

**Relevance**: Directly supports hypothesis that some hallucinations are recognizable but still produced due to generation dynamics.

---

### 3. INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection
**Authors**: Chen et al. (2024)
**Source**: arXiv:2402.03744, ICLR 2024
**File**: `papers/2402.03744_INSIDE.pdf`

**Key Contributions**:
- Proposes EigenScore: uses eigenvalues of sentence embedding covariance matrix
- Feature clipping approach for overconfident hallucinations
- State-of-the-art hallucination detection on CoQA, SQuAD, TriviaQA, NQ

**Critical Findings for Our Hypothesis**:
1. **Self-consistent hallucinations exist**: Some hallucinations are "overconfident" - consistent across samples
2. **Internal states encode truthfulness**: Dense semantic information in embeddings captures factuality
3. **Better LLMs = better detection**: LLaMA-13B enables better hallucination detection than 7B

**Methodology**:
- EigenScore = (1/K) * sum(log(λᵢ)) where λᵢ are eigenvalues of embedding covariance
- Lower EigenScore = more consistent responses = likely factual
- Feature clipping truncates extreme activations to reduce overconfidence

**Relevance**: Provides methods to identify "natural hallucinations" that are consistent and thus harder to detect via sampling.

---

### 4. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection
**Authors**: Manakul, Liusie, Gales (2023)
**Source**: arXiv:2303.08896, 685 citations
**File**: `papers/2303.08896_selfcheckgpt.pdf`

**Key Contributions**:
- Self-consistency based hallucination detection without external databases
- Multiple methods: BERTScore, QA, n-gram, NLI, LLM prompting
- Applicable to black-box models (no access to logits needed)

**Critical Findings for Our Hypothesis**:
1. **Factual = consistent**: If LLM knows a fact, sampled responses are similar and consistent
2. **Hallucinations = divergent**: Hallucinated facts show contradiction across samples
3. **Sentence-level detection**: Can identify specific hallucinated sentences within passages

**Key Insight**: Works well for "random" hallucinations but may struggle with consistent/"natural" hallucinations that transfer across samples.

---

### 5. Surveys on LLM Hallucination

#### A Survey on Hallucination in Large Language Models (Huang et al., 2023)
**Source**: arXiv:2311.05232, 1960 citations
**File**: `papers/2311.05232_hallucination_survey_huang.pdf`

**Taxonomy**:
- **Factuality hallucination**: Contradicts established facts
- **Faithfulness hallucination**: Diverges from source content/context

**Causes**:
1. Data: Misinformation in training data, knowledge boundary issues
2. Training: Exposure bias, knowledge shortcuts
3. Inference: Decoding strategies, context window limitations

#### Siren's Song in the AI Ocean (Zhang et al., 2023)
**Source**: arXiv:2309.01219, 819 citations
**File**: `papers/2309.01219_hallucination_siren_song.pdf`

Complements the above with analysis of detection and mitigation methods.

---

### 6. Additional Detection Methods

#### Semantic Entropy Probes (Kossen et al., 2024)
**Source**: arXiv:2406.15927
**File**: `papers/2406.15927_semantic_entropy_probes.pdf`

- Cheap probes trained on semantic entropy signals
- Uses internal representations rather than multiple generations
- More efficient than sampling-based methods

#### LLM Internal States Reveal Hallucination Risk (Ji et al., 2024)
**Source**: arXiv:2407.03282
**File**: `papers/2407.03282_internal_states_risk.pdf`

- Studies if LLMs can estimate their own hallucination risk
- Inspired by human self-awareness of knowledge gaps

---

### 7. Self-Awareness and Uncertainty

#### R-Tuning: Instructing LLMs to Say 'I Don't Know' (Zhang et al., 2023)
**Source**: arXiv:2311.09677
**File**: `papers/2311.09677_r_tuning.pdf`

- Fine-tunes models to refuse answering when uncertain
- Reduces hallucinations by encouraging appropriate abstention

#### SaySelf: Teaching LLMs to Express Confidence (Xu et al., 2024)
**Source**: arXiv:2405.20974
**File**: `papers/2405.20974_SaySelf.pdf`

- Self-reflective rationales for confidence expression
- Aims to calibrate verbal uncertainty with actual knowledge

---

## Common Methodologies

### Hallucination Detection Approaches

| Method | Type | Mechanism | Strengths | Limitations |
|--------|------|-----------|-----------|-------------|
| Self-consistency | Black-box | Multiple samples + similarity | No model access needed | Fails for consistent hallucinations |
| EigenScore | White-box | Eigenvalues of embedding covariance | Handles some overconfident cases | Requires internal access |
| Semantic entropy | White-box | Entropy over semantic clusters | Principled uncertainty | Computational cost |
| Trained probes | White-box | Classifiers on hidden states | Fast inference | Requires training data |
| LLM-as-judge | Black-box | Ask another LLM to verify | Flexible | Expensive, may share biases |

### Evaluation Metrics
- **AUROC**: Area under ROC curve for detection
- **Accuracy/Truthfulness rate**: Fraction of correct answers
- **Pearson correlation**: Correlation between detection score and factuality
- **AUC-PR**: For imbalanced datasets

---

## Standard Baselines

1. **Perplexity**: -1/T * sum(log p(yₜ))
2. **Length-normalized entropy**: Average token entropy
3. **Lexical similarity**: ROUGE-L across samples
4. **Token probability thresholds**: Max(-log p) or Avg(-log p)

---

## Datasets in the Literature

| Dataset | Size | Task | Used In |
|---------|------|------|---------|
| TruthfulQA | 817 Q | Truthfulness benchmark | TruthfulQA, many follow-ups |
| HaluEval | 35K | Hallucination detection | HaluEval paper, INSIDE |
| CoQA | 8K | Conversational QA | INSIDE, many detection papers |
| TriviaQA | 95K | Closed-book QA | INSIDE, many detection papers |
| Natural Questions | 307K | Open-domain QA | INSIDE, detection papers |
| SQuAD | 100K | Reading comprehension | INSIDE, detection papers |
| WikiBio (SelfCheckGPT) | 238 | Biography generation | SelfCheckGPT |

---

## Gaps and Opportunities

### What's Missing

1. **Systematic study of hallucination transfer**: No paper comprehensively studies which hallucinations transfer across model families
2. **Temporal persistence**: Limited work on whether hallucinations persist after additional training/RLHF
3. **Fine-grained categorization of "natural" hallucinations**: TruthfulQA identifies imitative falsehoods broadly but doesn't categorize by difficulty of recognition
4. **Cross-version analysis**: How do hallucinations change between model versions (e.g., GPT-3 → GPT-4)?

### Open Questions

1. Are there questions that cause hallucinations in ALL models regardless of architecture?
2. Can we predict which hallucinations will be "natural" (hard to recognize) vs "random"?
3. Does training on TruthfulQA reduce hallucinations or just cause memorization?

---

## Recommendations for Experiments

### Recommended Datasets
1. **Primary**: TruthfulQA (817 questions with categories)
2. **Secondary**: HaluEval QA samples (10K with labels), SelfCheckGPT WikiBio (sentence-level annotations)
3. **Supporting**: CoQA, TriviaQA, NQ for comparison with other methods

### Recommended Baselines
1. Self-consistency (SelfCheckGPT variants)
2. EigenScore (INSIDE)
3. Semantic entropy probes (if internal access available)
4. Perplexity and entropy baselines

### Recommended Metrics
1. Per-question hallucination rate across models (transfer analysis)
2. Recognition accuracy (can model identify its own error in isolation)
3. Robustness to paraphrasing
4. Correlation with model size/training

### Methodological Considerations
1. Use greedy decoding for consistency with TruthfulQA paper
2. Test multiple model families (GPT, LLaMA, Mistral, Claude)
3. Control for question category effects
4. Consider both generation and multiple-choice evaluation

---

## Key Takeaways for "Natural Hallucinations" Research

1. **Evidence for the hypothesis exists**: TruthfulQA's inverse scaling and snowballing paper both suggest hallucinations that models "know" are wrong but still produce

2. **Transfer is likely**: GPT-Neo/J shows same patterns as GPT-3 without adversarial filtering, suggesting training distribution similarities cause shared hallucinations

3. **Robustness confirmed**: Paraphrasing and temperature changes don't substantially affect these hallucinations

4. **Detection methods exist**: Combination of self-consistency (for random) and internal states (for overconfident) can identify hallucinations

5. **Gap to fill**: No systematic study of which specific TruthfulQA questions cause consistent hallucinations across ALL tested models and whether these persist through training
