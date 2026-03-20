# Resources Catalog

## Summary
All resources gathered for studying "natural hallucinations" in LLMs — hallucinations that are robust to perturbation, difficult to self-detect, and potentially transferable across models.

## Papers
Total papers downloaded: 19

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| CHOKE: Certain Hallucinations Overriding Known Evidence | Simhi et al. | 2025 | papers/2502.12964_CHOKE_hallucinations.pdf | Closest to "natural hallucinations" concept |
| LLMs Know More Than They Show | Orgad et al. | 2024 | papers/2410.02707_LLMs_know_more_than_they_show.pdf | Internal knowledge vs output discrepancy |
| Distinguishing Ignorance from Error | Simhi et al. | 2024 | papers/2410.22071_distinguishing_ignorance_from_error.pdf | HK+/HK- taxonomy, cross-model Jaccard |
| Hallucination, Monofacts, and Miscalibration | Miao & Kearns | 2025 | papers/2502.08666_hallucination_monofacts_miscalibration.pdf | Theoretical lower bound on hallucination |
| Hallucinate or Memorize | Niimi | 2025 | papers/2511.08877_hallucinate_or_memorize.pdf | Memorization-hallucination duality |
| Why Language Models Hallucinate | Kalai & Nachum | 2025 | papers/2509.04664_why_language_models_hallucinate_kalai.pdf | Statistical pressures theory |
| Hallucination is Inevitable | Xu et al. | 2024 | papers/2401.11817_hallucination_is_inevitable.pdf | Formal impossibility proof |
| TruthfulQA | Lin et al. | 2022 | papers/2109.07958_truthfulqa.pdf | Foundational benchmark, inverse scaling |
| Sources of Hallucination | McKenna et al. | 2023 | papers/2305.14552_sources_of_hallucination.pdf | Attestation & frequency biases |
| Cross-Model Consistency (Finch-Zk) | — | 2025 | papers/2508.14314_cross_model_consistency_finch_zk.pdf | Shared errors across models |
| Phenomenology of Hallucinations | Ruscio & Thompson | 2026 | papers/2603.13911_phenomenology_of_hallucinations.pdf | Internal uncertainty detection |
| LLMs Will Always Hallucinate | Banerjee et al. | 2024 | papers/2409.05746_llms_will_always_hallucinate.pdf | Structural inevitability |
| Banishing Hallucinations | Li et al. | 2024 | papers/2406.17642_banishing_hallucinations_rethinking_generalization.pdf | Generalization rethinking |
| Sycophancy in LLMs | — | 2024 | papers/2411.15287_sycophancy_in_llms.pdf | RLHF exacerbates sycophancy |
| Factual Misalignment Short/Long Form | — | 2025 | papers/2510.11218_factual_misalignment_short_long_form.pdf | Cross-format misalignment |
| Chain of Verification | — | 2023 | papers/2309.11495_chain_of_verification.pdf | Verification-based mitigation |
| HaluEval | Li et al. | 2023 | papers/2305.11747_halueval.pdf | Hallucination evaluation benchmark |
| Earth is Flat — Factual Errors | — | 2024 | papers/2401.00761_earth_is_flat_factual_errors.pdf | Systematic error patterns |
| Don't Hallucinate, Abstain | — | 2024 | papers/2402.00367_dont_hallucinate_abstain.pdf | Multi-LLM knowledge gaps |

## Datasets
Total datasets downloaded: 4

| Name | Source | Size | Task | Location |
|------|--------|------|------|----------|
| TruthfulQA | HuggingFace (truthfulqa/truthful_qa) | 817 questions | Truthfulness QA | datasets/truthfulqa/ |
| TriviaQA | HuggingFace (trivia_qa, rc.nocontext) | 17,944 validation | Factual QA | datasets/triviaqa/ |
| Natural Questions Open | HuggingFace (google-research-datasets/nq_open) | 91,535 total | Factual QA | datasets/natural_questions/ |
| HaluEval | HuggingFace (pminervini/HaluEval) | ~35K across configs | Hallucination detection | datasets/halueval/ |

See datasets/README.md for download instructions and loading code.

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| TruthfulQA | github.com/sylinrl/TruthfulQA | Benchmark evaluation | code/truthfulqa/ |
| LLMs Know More | github.com/technion-cs-nlp/LLMsKnow | Hidden state probing | code/llms_know/ |
| Hallucination Mitigation | github.com/technion-cs-nlp/hallucination-mitigation | HK+/HK- detection | code/hallucination_mitigation/ |
| LLM NLI Analysis | github.com/Teddy-Li/LLM-NLI-Analysis | Attestation bias analysis | code/llm_nli_analysis/ |
| Semantic Uncertainty | github.com/jlko/semantic_uncertainty | Semantic entropy computation | code/semantic_uncertainty/ |

See code/README.md for key entry points.

## Recommendations for Experiment Design

### Primary Datasets
1. **TruthfulQA**: 817 curated questions targeting systematic falsehoods. Cross-model results available for comparison. Best for testing transfer and robustness.
2. **TriviaQA (validation)**: 18K factual questions. Used by CHOKE and HK+/HK- papers for direct methodology comparison.

### Baseline Methods
1. Uncertainty-based detection (token probability, semantic entropy)
2. Linear probing on hidden states (from LLMs Know More)
3. Prompt-based mitigation (weak baseline per CHOKE results)
4. Cross-model consistency checking (from Finch-Zk)

### Evaluation Metrics
1. Hallucination persistence rate across prompt variants
2. Cross-model Jaccard similarity of hallucinated instances
3. Self-detection accuracy (can models identify their own natural hallucinations?)
4. Temporal prediction accuracy (older→newer model hallucination prediction)

### Code to Adapt/Reuse
1. **semantic_uncertainty**: For computing semantic entropy and uncertainty measures
2. **hallucination_mitigation**: For HK+/HK- classification framework (WACK)
3. **llms_know**: For hidden state probing methodology
4. **truthfulqa**: For benchmark evaluation pipeline
