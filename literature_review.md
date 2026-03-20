# Literature Review: Natural Hallucinations in Large Language Models

## Research Area Overview

This review examines systematic, robust hallucinations in large language models (LLMs) — what we term "natural hallucinations." The research hypothesis posits that certain LLM hallucinations are: (1) robust to additional information or question reformulation, (2) difficult for LLMs to self-detect even with all necessary information present, (3) transferable across models, and (4) more easily memorized if encountered during training. We surveyed 19 papers spanning hallucination characterization, detection, mitigation, and theoretical foundations.

## Key Papers

### Paper 1: CHOKE — Certain Hallucinations Overriding Known Evidence
**Simhi, Itzhak, Barez, Stanovsky, Belinkov (2025)** — arXiv:2502.12964

- **Key Contribution**: Defines CHOKE as cases where LLMs demonstrably know the correct answer yet produce hallucinated responses with *high certainty* under trivial prompt perturbations. This is the closest existing concept to "natural hallucinations."
- **Methodology**: Three-stage pipeline: (1) knowledge test via few-shot prompting, (2) hallucination elicitation via 56 natural prompt variants, (3) certainty measurement via token probability, probability difference, and semantic entropy.
- **Datasets**: TriviaQA, Natural Questions (up to 20K examples per model).
- **Models**: Mistral-7B, Llama-3.1-8B, Gemma-2-9B/27B (base and instruct).
- **Key Results**: 16-43% of hallucinations-despite-knowledge occur with high certainty. CHOKE examples show significantly higher cross-prompt consistency (Jaccard 40.6% vs 13.6% random). Prompt-based mitigation largely fails (6-8% success); trained probes achieve 54% CHOKE detection.
- **Code**: Uses semantic_uncertainty repo. No standalone release.
- **Relevance**: Directly demonstrates robustness of certain hallucinations across prompts. The phenomenon appears across all tested model families but specific instances were not compared cross-model.

### Paper 2: LLMs Know More Than They Show
**Orgad, Toker, Gekhman et al. (2024)** — arXiv:2410.02707, ICLR 2025

- **Key Contribution**: Internal LLM representations encode truthfulness information at exact answer tokens, yet models consistently generate incorrect answers. Introduces error taxonomy including "consistently incorrect" (Type C) errors — the model gives the same wrong answer 29/30 times.
- **Methodology**: Linear probing classifiers on intermediate representations. Systematic probing across all layers and token positions.
- **Datasets**: 10 datasets including TriviaQA, HotpotQA, Natural Questions, Winobias, MNLI, Math, IMDB.
- **Models**: Mistral-7b, Llama3-8b (base and instruct variants).
- **Key Results**: Answer-token probing achieves AUC 0.85-0.95. Cross-task generalization is poor — truthfulness encoding is skill-specific. Using probes to select from resampled responses improves accuracy 30-40 points for Type C errors.
- **Code**: https://github.com/technion-cs-nlp/LLMsKnow
- **Relevance**: Type C errors (consistently incorrect despite internal knowledge) are the mechanistic signature of natural hallucinations.

### Paper 3: Distinguishing Ignorance from Error (HK+/HK-)
**Simhi, Herzig, Szpektor, Belinkov (2024)** — arXiv:2410.22071

- **Key Contribution**: Formalizes HK- (ignorance — model lacks knowledge) vs HK+ (error despite knowledge). Develops WACK framework for model-specific hallucination datasets.
- **Methodology**: Knowledge detection via multi-sample generation. HK+ elicitation via 4 settings (truthful, persona, Alice-Bob, snowballing). Detection via linear SVM on hidden states.
- **Datasets**: TriviaQA, Natural Questions (30K examples each).
- **Models**: Mistral-7B-v0.3, Llama-3.1-8B, Gemma-2-9B.
- **Key Results**: HK+ prevalence: 4-24%. **Cross-model HK+ Jaccard similarity only 0.10-0.36** (vs. 0.6-0.8 for knowledge). Prompt mitigation helps HK+ (9-51%) but barely helps HK- (0.8-3.2%).
- **Code**: https://github.com/technion-cs-nlp/hallucination-mitigation
- **Relevance**: HK+ is precisely natural hallucinations. Low cross-model Jaccard is **counterevidence** to transfer — different models hallucinate on different questions despite shared knowledge.

### Paper 4: Hallucination, Monofacts, and Miscalibration
**Miao & Kearns (2025)** — arXiv:2502.08666, PNAS 2026

- **Key Contribution**: Calibrated LLMs must hallucinate at a rate lower-bounded by monofact rate minus miscalibration. Selective upweighting of 5% of training data reduces hallucination by 40%.
- **Methodology**: Bigram models and SFT on T5/GPT2 with controlled fact frequency distributions.
- **Datasets**: IMDb movie facts, synthetic biographies (10K).
- **Models**: Bigram, T5-Small/Large, GPT2-Medium/Large.
- **Key Results**: Hallucination scales linearly with monofact rate. Architecture-independent. Upweighting 312 examples (~6%) can halve hallucination.
- **Relevance**: Theoretical grounding — natural hallucinations are statistically inevitable for rare facts. Architecture-independence supports cross-model transfer of the phenomenon.

### Paper 5: Hallucinate or Memorize
**Niimi (2025)** — arXiv:2511.08877

- **Key Contribution**: Hallucination and memorization are two sides of the same probabilistic process. Citation count proxies training frequency. Inflection at ~90 citations; saturation at ~1,248.
- **Key Results**: High-citation group scored 72% higher on accuracy (Cohen's d = 1.02). Log(citations) vs. similarity: r = 0.75, R² = 0.56.
- **Relevance**: Directly supports hypothesis that hallucinations used in training could be more easily memorized.

### Paper 6: TruthfulQA
**Lin, Hilton, Evans (2022)** — arXiv:2109.07958

- **Key Contribution**: Benchmark of 817 questions eliciting "imitative falsehoods." Larger models are *less* truthful (inverse scaling).
- **Datasets**: 817 questions, 38 categories. https://github.com/sylinrl/TruthfulQA
- **Models**: GPT-3, GPT-Neo/J, GPT-2, UnifiedQA, plus external evaluations.
- **Key Results**: Best model 58% truthful vs 94% human. Inverse scaling across 4 model families. Persistent across paraphrases.
- **Relevance**: Cross-model consistency of imitative falsehoods strongly supports transfer hypothesis. Robustness across paraphrases supports persistence.

### Paper 7: Sources of Hallucination on Inference Tasks
**McKenna, Li et al. (2023)** — arXiv:2305.14552

- **Key Contribution**: Attestation bias (models affirm training-attested hypotheses regardless of premise) and relative frequency bias drive NLI hallucinations.
- **Models**: LLaMA-65B, GPT-3.5, PaLM-540B, GPT-4.
- **Key Results**: 1.9-2.2x more false entailment for attested hypotheses. GPT-3.5 recall drops 92.3→55.3 with entity replacement. Biases resist prompt engineering.
- **Code**: https://github.com/Teddy-Li/LLM-NLI-Analysis
- **Relevance**: Cross-model consistency across LLaMA, GPT-3.5, PaLM, GPT-4 demonstrates training-data-driven hallucinations transfer. Resistance to prompting shows robustness.

### Additional Relevant Papers

| Paper | ID | Year | Key Finding |
|-------|-----|------|------------|
| Cross-Model Consistency Finch-Zk | 2508.14314 | 2025 | Cross-model consistency reinforces shared errors |
| Phenomenology of Hallucinations | 2603.13911 | 2026 | Models detect uncertainty internally but signal silent at output |
| Hallucination is Inevitable | 2401.11817 | 2024 | Formal proof LLMs cannot avoid hallucination |
| Why Language Models Hallucinate (Kalai) | 2509.04664 | 2025 | Natural statistical pressures cause systematic hallucination |
| LLMs Will Always Hallucinate | 2409.05746 | 2024 | Structural inevitability via Gödel's theorem |
| Banishing Hallucinations | 2406.17642 | 2024 | Next-token prediction hallucinations are threshold-based |
| Sycophancy in LLMs | 2411.15287 | 2024 | RLHF exacerbates sycophancy |
| Factual Misalignment Short/Long Form | 2510.11218 | 2025 | Systematic misalignment across 16 LLMs by query format |
| Chain of Verification | 2309.11495 | 2023 | Verification reduces hallucination |
| HaluEval | 2305.11747 | 2023 | Hallucination evaluation benchmark |
| Earth is Flat — Factual Errors | 2401.00761 | 2024 | Systematic factual error patterns |
| Don't Hallucinate, Abstain | 2402.00367 | 2024 | Multi-LLM collaboration for knowledge gaps |

## Common Methodologies

1. **Knowledge probing**: Few-shot prompting with multiple samples to establish whether a model "knows" the answer (CHOKE, HK+/HK-)
2. **Linear probing on hidden states**: Classifiers on intermediate representations to detect truthfulness (LLMs Know More, HK+/HK-)
3. **Prompt perturbation**: Test hallucination persistence across paraphrases and prompt variants (CHOKE, TruthfulQA)
4. **Cross-model evaluation**: Test same questions across model families (all papers)
5. **Semantic entropy**: Cluster generations semantically to measure uncertainty (CHOKE)

## Standard Baselines

- TruthfulQA benchmark (817 questions, many published results)
- Few-shot QA on TriviaQA / Natural Questions
- Prompt-based mitigation ("be truthful" instructions — consistently weak)
- Uncertainty-based detection (token probabilities, semantic entropy)
- Linear probes on hidden states

## Evaluation Metrics

- **Truthfulness rate**: Fraction of factually correct responses
- **CHOKE-Score**: Detection rate for high-certainty hallucinations
- **AUC**: Binary classification of hallucinated vs correct
- **Jaccard similarity**: Overlap of hallucinated examples across models/prompts
- **Exact match**: Factual QA evaluation
- **Semantic entropy**: Uncertainty quantification

## Datasets in the Literature

| Dataset | Used By | Task | Size |
|---------|---------|------|------|
| TruthfulQA | TruthfulQA, many evaluations | Truthfulness QA | 817 questions |
| TriviaQA | CHOKE, HK+/HK-, LLMs Know | Factual QA | 95K+ |
| Natural Questions | CHOKE, HK+/HK-, LLMs Know | Factual QA | 91K+ |
| HaluEval | Various evaluations | Hallucination detection | ~35K |
| Levy/Holt | Sources of Hallucination | NLI | 2.4K |

## Gaps and Opportunities

1. **No unified "natural hallucinations" study**: No paper systematically studies all four aspects (robustness, self-detection difficulty, cross-model transfer, memorization predictability) together.
2. **Cross-model transfer is contested**: TruthfulQA and Sources of Hallucination show cross-model patterns, but HK+/HK- shows low Jaccard (0.10-0.36) for specific instances.
3. **Temporal prediction unexplored**: No paper tests whether hallucinations in older models predict hallucinations in newer models of the same family.
4. **Training data frequency link underexplored**: Monofacts theory validated only on small models.
5. **Self-detection of robust hallucinations**: CHOKE shows uncertainty methods fail, probing shows promise. Gap between internal knowledge and output remains unexplained.

## Recommendations for Our Experiment

### Recommended Datasets
1. **TruthfulQA** (primary): 817 questions, cross-model results for comparison
2. **TriviaQA** (validation, 18K): Used by CHOKE and HK+/HK- papers
3. **Natural Questions** (open-domain, 3.6K validation): Complementary factual QA

### Recommended Approach
1. **Identify natural hallucination candidates**: CHOKE-style — test across prompt variants, find high-certainty hallucinations despite knowledge
2. **Test cross-model transfer**: Same instances across model families, measure Jaccard overlap
3. **Test temporal prediction**: Compare across model versions (Llama-2 vs Llama-3, etc.)
4. **Test robustness**: Persistence under information augmentation (providing correct answer in context)

### Recommended Baselines
- Random baseline
- Uncertainty-based detection (token probability, semantic entropy)
- Linear probing on hidden states
- Prompt-based mitigation

### Recommended Metrics
- Hallucination persistence rate (fraction surviving perturbation)
- Cross-model Jaccard similarity
- Self-detection accuracy
- Temporal prediction accuracy

### Methodological Considerations
- Use multiple prompt variants (56+ per CHOKE) for reliable robustness measurement
- Control for HK- vs HK+ — natural hallucinations should be HK+
- Consider monofact theory — natural hallucinations may cluster around rare facts
- Account for model-specific vs universal patterns
