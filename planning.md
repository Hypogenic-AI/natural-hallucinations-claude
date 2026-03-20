# Research Plan: Natural Hallucinations in Large Language Models

## Motivation & Novelty Assessment

### Why This Research Matters

LLMs still hallucinate frequently despite advances in scale and training. The user hypothesis suggests a compelling reason: some hallucinations are "natural" in that they are difficult for models to recognize even when all necessary information is available. If certain hallucinations are robust to additional context, transfer across model families, and persist through training, this would fundamentally change our understanding of how to mitigate hallucinations. Rather than just adding more knowledge, we may need to address systematic biases in how LLMs process information.

### Gap in Existing Work

Based on the literature review:
1. **TruthfulQA (2021)** identifies "imitative falsehoods" and shows inverse scaling, but doesn't systematically study which specific questions cause consistent errors across ALL models
2. **Snowballing paper (2023)** shows models can recognize their errors in isolation, but focuses on sequential reasoning errors, not the persistent "natural hallucinations"
3. **No paper comprehensively studies**: (a) which hallucinations transfer across model families, (b) whether hallucinations found in older models predict those in newer ones, and (c) whether these errors are robust to question rephrasing

### Our Novel Contribution

We will systematically identify "natural hallucinations" - questions where multiple diverse models consistently produce the same incorrect answer - and characterize their properties:
1. **Transfer**: Do the same questions cause hallucinations across GPT-4, Claude, and open-source models?
2. **Robustness**: Are these hallucinations robust to paraphrasing the question?
3. **Self-Recognition**: Can models recognize these as errors when asked directly, or are they "blindspots"?

### Experiment Justification

- **Experiment 1 (Cross-Model Transfer)**: Tests whether hallucinations transfer across model families - essential for understanding if hallucinations are model-specific or arise from common training data/objectives
- **Experiment 2 (Robustness to Paraphrasing)**: Tests whether "natural hallucinations" persist regardless of how questions are phrased - crucial for understanding if these are superficial pattern matches or deep misconceptions
- **Experiment 3 (Self-Recognition)**: Tests whether models can identify their own errors when presented separately - determines if these are "unknowable" to the model or just contextually triggered

---

## Research Question

Are there "natural hallucinations" - factual errors that (1) multiple LLMs consistently produce, (2) persist despite rephrasing questions, and (3) models fail to recognize as errors even when explicitly asked? If so, do hallucinations in older models predict hallucinations in newer models?

## Background and Motivation

The TruthfulQA benchmark demonstrates that LLMs produce "imitative falsehoods" - answers that match common human misconceptions in training data. However, we don't know:
1. Which specific questions cause consistent errors across different model families
2. Whether these errors are robust to question variations
3. Whether models have any capability to recognize these errors

Understanding these questions is critical because:
- If hallucinations transfer, they likely stem from common training data biases
- If they're robust to paraphrasing, simple prompt engineering won't solve them
- If models can't recognize them, self-critique approaches will fail

## Hypothesis Decomposition

**H1**: Some hallucinations transfer across model families
- Test: Compare error patterns across GPT-4o, Claude 3.5 Sonnet, and Llama 3 70B on TruthfulQA
- Success criteria: Significant correlation in per-question error rates across models

**H2**: Transferred hallucinations are robust to question paraphrasing
- Test: For questions with high cross-model error rates, rephrase and re-test
- Success criteria: Error rate remains high (>70%) on paraphrased versions

**H3**: Models fail to recognize "natural hallucinations" as errors
- Test: Present the model's own wrong answer and ask if it's correct
- Success criteria: Recognition rate is significantly lower for "natural" vs "random" hallucinations

**H4**: Older model hallucinations predict newer model hallucinations
- Test: Compare error patterns between older (GPT-3.5) and newer (GPT-4o) models
- Success criteria: Questions causing errors in older models have higher error rates in newer models

## Proposed Methodology

### Approach

We will use the TruthfulQA benchmark (817 questions) and test multiple LLMs through their APIs. We'll identify "natural hallucinations" as questions where:
1. Multiple models (≥3 out of 4 tested) give incorrect answers
2. The errors are similar across models (not random)

Then we'll characterize these natural hallucinations by:
1. Testing robustness to paraphrasing
2. Testing self-recognition ability
3. Comparing older vs newer model performance

### Experimental Steps

**Step 1: Cross-Model TruthfulQA Evaluation** (Exp 1)
- Run GPT-4o, Claude 3.5 Sonnet, Llama 3 70B, and GPT-3.5-turbo on TruthfulQA
- Use greedy decoding (temperature=0) for reproducibility
- Score answers using the GPT-judge approach from TruthfulQA paper
- Identify questions with high cross-model error rates

**Step 2: Identify Natural Hallucinations**
- Calculate per-question error rate across models
- Define "natural hallucinations" as questions where ≥3/4 models fail
- Categorize by TruthfulQA category (Misconceptions, Conspiracies, etc.)

**Step 3: Robustness to Paraphrasing** (Exp 2)
- Take top 50 natural hallucinations
- Generate 3 paraphrases per question using Claude
- Test original models on paraphrased versions
- Calculate robustness score: % of paraphrases that still produce errors

**Step 4: Self-Recognition Test** (Exp 3)
- For questions where model X gave wrong answer A
- Ask model X: "Is the following answer correct? [A]"
- Compare recognition rate for:
  - Natural hallucinations (cross-model errors)
  - Random hallucinations (model-specific errors)

**Step 5: Temporal Analysis** (Exp 4)
- Compare GPT-3.5 vs GPT-4o error patterns
- Test if GPT-3.5 errors predict GPT-4o errors
- Calculate predictive power using correlation and chi-squared tests

### Baselines

1. **Random baseline**: Expected error rate if hallucinations were independent across models
2. **Category baseline**: Error rate expected from TruthfulQA category difficulty
3. **Model-specific baseline**: Per-model error rate for calculating excess cross-model correlation

### Evaluation Metrics

1. **Cross-Model Error Correlation**: Pearson correlation of per-question error rates between model pairs
2. **Natural Hallucination Rate**: % of questions where ≥3/4 models fail
3. **Robustness Score**: % of paraphrased questions that still produce errors
4. **Self-Recognition Accuracy**: % of own errors model can identify
5. **Temporal Prediction**: Correlation between older and newer model errors

### Statistical Analysis Plan

- **Correlation tests**: Pearson correlation with bootstrapped 95% CIs
- **Difference tests**: McNemar's test for paired proportions (same questions, different conditions)
- **Multiple comparisons**: Benjamini-Hochberg FDR correction
- **Significance level**: α = 0.05
- **Effect size reporting**: Cohen's d for comparisons, correlation coefficient interpretation

## Expected Outcomes

**If H1 supported**: High cross-model correlation (r > 0.5) on per-question errors, suggesting shared training data biases cause systematic hallucinations

**If H2 supported**: Robustness scores > 70% for natural hallucinations, indicating deep misconceptions rather than surface pattern matching

**If H3 supported**: Self-recognition accuracy significantly lower for natural vs random hallucinations, revealing "blindspots"

**If H4 supported**: Significant predictive power of older model errors for newer model errors, useful for anticipating future LLM failures

## Timeline and Milestones

1. **Phase 1** - Environment Setup & Data Loading (15 min)
2. **Phase 2** - Cross-Model Evaluation (60 min, API calls)
3. **Phase 3** - Natural Hallucination Identification & Analysis (20 min)
4. **Phase 4** - Paraphrasing Robustness Test (30 min)
5. **Phase 5** - Self-Recognition Test (20 min)
6. **Phase 6** - Temporal Analysis & Final Stats (20 min)
7. **Phase 7** - Documentation & Report (30 min)

## Potential Challenges

1. **API costs/rate limits**: Mitigate by starting with subset (100-200 questions), scaling if promising
2. **Judge accuracy**: Use both automated GPT-judge and manual spot-checking
3. **Paraphrase quality**: Generate diverse paraphrases and filter for semantic equivalence
4. **Model version changes**: Document exact model versions and timestamps

## Success Criteria

The research is successful if we can:
1. Identify ≥20 "natural hallucinations" that appear across ≥3 models
2. Demonstrate these hallucinations have robustness scores > 50%
3. Show self-recognition is significantly impaired for natural vs random hallucinations
4. Provide actionable insights for hallucination mitigation

## Resources

### Models (via OpenAI API)
- GPT-4.1 (OpenAI) - latest state-of-the-art
- GPT-4o (OpenAI) - previous generation SOTA
- GPT-4o-mini (OpenAI) - smaller/cheaper model
- GPT-3.5-turbo (OpenAI) - older baseline for temporal analysis

Note: Only OpenAI API key available. Cross-model transfer tested across
model sizes and generations within OpenAI family. While not cross-family,
this tests temporal prediction (older→newer) and size effects effectively.

### Dataset
- TruthfulQA (817 questions, 38 categories) - primary
- Already downloaded to `datasets/truthfulqa/`

### Code to Adapt
- `code/TruthfulQA/` - evaluation harness and GPT-judge
- `code/snowball_hallucination/` - self-recognition methodology
