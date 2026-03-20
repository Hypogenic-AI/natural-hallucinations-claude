# Code Repositories for "Natural Hallucinations" Research

Cloned with `git clone --depth 1` on 2026-03-20.

---

## 1. TruthfulQA

- **Path:** `truthfulqa/`
- **Source:** https://github.com/sylinrl/TruthfulQA
- **Purpose:** Benchmark of 817 questions measuring whether LLMs mimic human falsehoods. Supports generation and multiple-choice (MC1/MC2) evaluation tasks.
- **Key files:**
  - `TruthfulQA.csv` -- Full benchmark dataset (questions + reference answers)
  - `truthfulqa/evaluate.py` -- Main evaluation script
  - `truthfulqa/metrics.py` -- Scoring metrics (BLEURT, ROUGE, BLEU, GPT-judge)
  - `truthfulqa/models.py` -- Model interface for evaluation

---

## 2. LLMs Know More Than They Show

- **Path:** `llms_know/`
- **Source:** https://github.com/technion-cs-nlp/LLMsKnow
- **Purpose:** Probing LLM internal representations to detect hallucinations. Shows that LLMs encode correctness, error type, and question difficulty internally even when producing wrong answers.
- **Key files:**
  - `src/generate_model_answers.py` -- Generate model answers per dataset
  - `src/extract_exact_answer.py` -- Extract exact answers from model outputs
  - `src/probe_all_layers_and_tokens.py` -- Probe all layers/tokens for heatmaps (Section 2)
  - `src/probe.py` -- Probe specific layer and token (Section 3)
  - `src/probe_choose_answer.py` -- Use probes to select better answers
  - `src/logprob_detection.py` -- Log-probability based hallucination detection
  - `src/probe_type_of_error.py` -- Classify error types via probing
- **Supported models:** Mistral-7B (v0.2 Instruct, v0.3), Llama-3-8B (base + Instruct)

---

## 3. Hallucination Mitigation (HK+/HK- Detection)

- **Path:** `hallucination_mitigation/`
- **Source:** https://github.com/technion-cs-nlp/hallucination-mitigation
- **Purpose:** Two papers: (a) benchmarks and interventions for combating hallucinations (attention-based interventions, dynamic intervention); (b) distinguishing ignorance-based vs. error-based hallucinations using WACK datasets.
- **Key files:**
  - `Constructing_Benchmarks_and_Interventions_for_Combating_Hallucinations_in_LLMs/`
    - `RunAllSteps.py` -- End-to-end pipeline
    - `ModelInside.py` -- Internal model analysis
    - `InteventionByDetection.py` -- Intervention methods
    - `DatasetCreationWithoutContext.py` -- Dataset generation
  - `Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations/`
    - `RunAllSteps.py` -- End-to-end pipeline
    - `ModelInside.py` -- Internal model analysis for error vs ignorance
    - `plot_results.py` -- Visualization

---

## 4. LLM NLI Analysis (Sources of Hallucination)

- **Path:** `llm_nli_analysis/`
- **Source:** https://github.com/Teddy-Li/LLM-NLI-Analysis
- **Purpose:** Investigates sources of hallucination in NLI tasks, focusing on attestation bias and frequency priors. Uses Levy/Holt directional entailment dataset.
- **Key files:**
  - `gpt3_inference.py` -- Run NLI with GPT-3/4 models
  - `llama_inference.py` -- Run NLI with LLaMA models
  - `poll_attestation.py` -- Measure attestation bias
  - `get_frequencies_ngram.py` -- Measure relative frequency prior
  - `randprem_experiments.py` -- Random premise controlled experiments
  - `frequency_controlled_experiments.py` -- Frequency-controlled evaluation

---

## 5. Semantic Uncertainty

- **Path:** `semantic_uncertainty/`
- **Source:** https://github.com/jlko/semantic_uncertainty
- **Purpose:** Detecting hallucinations via semantic entropy -- clusters sampled LLM responses by meaning and computes entropy over semantic clusters. Published in Nature.
- **Key files:**
  - `semantic_uncertainty/generate_answers.py` -- Generate LLM answers with sampling
  - `semantic_uncertainty/compute_uncertainty_measures.py` -- Compute semantic entropy and other uncertainty measures
  - `semantic_uncertainty/analyze_results.py` -- Analyze and aggregate results
  - `semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py` -- Core semantic entropy implementation
  - `semantic_uncertainty/uncertainty/uncertainty_measures/p_true.py` -- P(True) baseline
  - `semantic_uncertainty/uncertainty/models/huggingface_models.py` -- HuggingFace model wrappers
