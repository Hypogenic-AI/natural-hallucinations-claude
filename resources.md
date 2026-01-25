# Resources Catalog for Natural Hallucinations Research

## Summary

This document catalogs all resources gathered for studying "natural hallucinations" in large language models - hallucinations that are difficult for models to recognize even when relevant information is present.

**Total papers downloaded**: 35
**Total datasets downloaded**: 6
**Total repositories cloned**: 5

---

## Papers

### Core Papers (Deep Read)

| Title | Authors | Year | File | Key Contribution |
|-------|---------|------|------|------------------|
| TruthfulQA | Lin, Hilton, Evans | 2021 | `papers/2109.07958_TruthfulQA.pdf` | Benchmark for imitative falsehoods; inverse scaling |
| Hallucination Snowball | Zhang et al. | 2023 | `papers/2305.13534_hallucination_snowball.pdf` | LLMs recognize 67-87% of own errors separately |
| INSIDE | Chen et al. | 2024 | `papers/2402.03744_INSIDE.pdf` | EigenScore for internal state detection |
| SelfCheckGPT | Manakul et al. | 2023 | `papers/2303.08896_selfcheckgpt.pdf` | Self-consistency detection method |

### Survey Papers

| Title | Authors | Year | File | Scope |
|-------|---------|------|------|-------|
| Hallucination Survey | Huang et al. | 2023 | `papers/2311.05232_hallucination_survey_huang.pdf` | Comprehensive taxonomy and methods |
| Siren's Song | Zhang et al. | 2023 | `papers/2309.01219_hallucination_siren_song.pdf` | Detection and mitigation overview |
| LLM Evaluation Survey | Chang et al. | 2023 | `papers/2307.03109_llm_eval_survey_comprehensive.pdf` | Broad evaluation methods |

### Detection Methods

| Title | Authors | Year | File | Method |
|-------|---------|------|------|--------|
| Semantic Entropy Probes | Kossen et al. | 2024 | `papers/2406.15927_semantic_entropy_probes.pdf` | Probes on semantic entropy |
| Internal States Risk | Ji et al. | 2024 | `papers/2407.03282_internal_states_risk.pdf` | Self-awareness of risk |
| SAC3 | Zhang et al. | 2023 | `papers/2311.01740_SAC3.pdf` | Semantic-aware cross-check |
| Attention Satisfies | Yuksekgonul et al. | 2023 | `papers/2309.15098_attention_satisfies.pdf` | Constraint satisfaction view |
| Cross Examination | Cohen et al. | 2023 | `papers/2305.13281_cross_examination.pdf` | LM vs LM detection |
| InterrogateLLM | Yehuda et al. | 2024 | `papers/2403.02889_interrogate_llm.pdf` | Zero-resource interrogation |
| Cost-Effective Detection | Valentin et al. | 2024 | `papers/2407.21424_cost_effective_hallu.pdf` | Production-focused pipeline |
| Steer Latents | Park et al. | 2025 | `papers/2503.01917_steer_latents.pdf` | Latent space steering |
| HaloScope | Du et al. | 2024 | `papers/2409.17504_HaloScope.pdf` | Unlabeled generation detection |

### Mitigation and Calibration

| Title | Authors | Year | File | Approach |
|-------|---------|------|------|----------|
| TruthX | Zhang et al. | 2024 | `papers/2402.17811_TruthX.pdf` | Editing in truthful space |
| Self-Alignment | Zhang et al. | 2024 | `papers/2402.09267_self_alignment_factuality.pdf` | Self-evaluation alignment |
| R-Tuning | Zhang et al. | 2023 | `papers/2311.09677_r_tuning.pdf` | Teaching "I don't know" |
| SaySelf | Xu et al. | 2024 | `papers/2405.20974_SaySelf.pdf` | Confidence expression |
| Verbal Uncertainty | Ji et al. | 2025 | `papers/2503.14477_verbal_uncertainty.pdf` | Linear uncertainty features |
| SLED | Zhang et al. | 2024 | `papers/2411.02433_SLED.pdf` | Self logits evolution |
| Sharpness Alerts | Chen et al. | 2024 | `papers/2403.01548_sharpness_alerts.pdf` | Inner representation view |

### Benchmarks and Evaluation

| Title | Authors | Year | File | Benchmark |
|-------|---------|------|------|-----------|
| Dawn After Dark | Li et al. | 2024 | `papers/2401.03205_dawn_after_dark.pdf` | Factuality hallucination study |
| Fine-grained Detection | Mishra et al. | 2024 | `papers/2401.06855_fine_grained_hallucination.pdf` | Taxonomy and editing |
| FactBench | Muhlgay et al. | 2023 | `papers/2307.06908_factbench.pdf` | Factuality benchmark generation |
| Factuality Enhanced | Lee et al. | 2022 | `papers/2206.04624_factuality_enhanced.pdf` | FactualityPrompts |
| FactKB | Feng et al. | 2023 | `papers/2305.08281_FactKB.pdf` | Knowledge-enhanced evaluation |
| HalluLens | Bang et al. | 2025 | `papers/2504.17550_HalluLens.pdf` | Comprehensive benchmark |
| RAGTruth | Wu et al. | 2023 | `papers/2401.00396_RAGTruth.pdf` | RAG hallucination corpus |
| SimpleQA | | 2025 | `papers/2509.07968_simpleqa.pdf` | Factuality benchmark |
| ReFACT | Wang et al. | 2025 | `papers/2509.25868_refact.pdf` | Scientific confabulation |

### Domain-Specific

| Title | Authors | Year | File | Domain |
|-------|---------|------|------|--------|
| Legal Hallucinations | Dahl et al. | 2024 | `papers/2401.01301_legal_hallucinations.pdf` | Legal domain |
| MedHalu | Agarwal et al. | 2024 | `papers/2409.19492_medhalu.pdf` | Healthcare queries |
| Entity Hallucination | | 2025 | `papers/2502.11948_entity_hallucination.pdf` | Entity-level analysis |

See `papers/README.md` for full paper descriptions.

---

## Datasets

### Downloaded Datasets

| Name | Source | Size | Location | Primary Use |
|------|--------|------|----------|-------------|
| TruthfulQA | `truthfulqa/truthful_qa` | 817 Q | `datasets/truthfulqa/` | Imitative falsehoods benchmark |
| HaluEval | `pminervini/HaluEval` | 10K | `datasets/halueval/` | Hallucination detection |
| SelfCheckGPT WikiBio | `potsawee/wiki_bio_gpt3_hallucination` | 238 | `datasets/selfcheckgpt_wikibio/` | Sentence-level detection |
| TriviaQA | `trivia_qa` | 1K subset | `datasets/triviaqa/` | Factual QA |
| NQ Open | `nq_open` | 1K subset | `datasets/nq_open/` | Open-domain QA |
| CoQA | `stanfordnlp/coqa` | 500 subset | `datasets/coqa/` | Conversational QA |

See `datasets/README.md` for download instructions and detailed descriptions.

---

## Code Repositories

| Name | URL | Location | Purpose |
|------|-----|----------|---------|
| TruthfulQA | github.com/sylinrl/TruthfulQA | `code/TruthfulQA/` | Official benchmark implementation |
| SelfCheckGPT | github.com/potsawee/selfcheckgpt | `code/selfcheckgpt/` | Self-consistency detection |
| INSIDE/EigenScore | github.com/alibaba/eigenscore | `code/INSIDE_eigenscore/` | Internal state detection |
| Snowball Hallucination | github.com/Nanami18/Snowballed_Hallucination | `code/snowball_hallucination/` | Snowballing analysis |
| HaluEval | github.com/RUCAIBox/HaluEval | `code/HaluEval/` | Evaluation benchmark |

See `code/README.md` for usage instructions.

---

## Resource Gathering Notes

### Search Strategy
1. Paper-finder service with diligent mode for initial search
2. Multiple queries targeting different aspects:
   - "LLM hallucinations detection recognition"
   - "LLM hallucination robustness transfer across models"
   - "factual error detection language models benchmarks"
   - "LLM self-awareness uncertainty calibration hallucination"
   - "LLM benchmark TruthfulQA factual questions dataset"
3. ArXiv direct downloads for specific papers

### Selection Criteria
- Prioritized papers with >50 citations or published in top venues (ICLR, ACL, EMNLP)
- Focus on papers studying detection, robustness, and transfer of hallucinations
- Included foundational benchmarks (TruthfulQA) and latest methods (INSIDE, 2024)

### Challenges Encountered
- Some datasets not publicly available (RAGTruth full corpus)
- Some papers had incorrect arXiv IDs requiring title-based search
- Large datasets (full TriviaQA, NQ) downloaded as subsets to manage size

---

## Recommendations for Experiment Design

### Primary Datasets
1. **TruthfulQA** - Core benchmark for imitative falsehoods
2. **HaluEval** - For training/evaluating detection methods

### Baseline Methods
1. Self-consistency (SelfCheckGPT)
2. EigenScore (INSIDE)
3. Perplexity and entropy

### Evaluation Strategy
1. Test multiple model families (GPT, LLaMA, Mistral)
2. Measure per-question consistency of hallucinations across models
3. Test if models can recognize their own hallucinations in isolation
4. Evaluate robustness to paraphrasing

### Code to Adapt/Reuse
1. **TruthfulQA repo**: Evaluation framework and GPT-judge
2. **Snowball repo**: Verification methodology for self-recognition
3. **INSIDE repo**: EigenScore implementation for internal state analysis

---

## File Locations Summary

```
natural-hallucinations-claude/
├── papers/                          # Downloaded PDFs
│   ├── README.md                    # Paper list with descriptions
│   ├── pages/                       # Chunked PDFs for reading
│   ├── search_results_*.txt         # Search outputs
│   └── *.pdf                        # Paper files
├── datasets/                        # Downloaded datasets
│   ├── README.md                    # Download instructions
│   ├── .gitignore                   # Excludes data files
│   ├── truthfulqa/                  # TruthfulQA data
│   ├── halueval/                    # HaluEval data
│   ├── selfcheckgpt_wikibio/        # WikiBio annotations
│   ├── triviaqa/                    # TriviaQA subset
│   ├── nq_open/                     # NQ subset
│   └── coqa/                        # CoQA subset
├── code/                            # Cloned repositories
│   ├── README.md                    # Usage instructions
│   ├── TruthfulQA/                  # Official benchmark
│   ├── selfcheckgpt/                # Detection method
│   ├── INSIDE_eigenscore/           # Internal state method
│   ├── snowball_hallucination/      # Snowballing analysis
│   └── HaluEval/                    # Evaluation benchmark
├── literature_review.md             # Synthesis of findings
├── resources.md                     # This file
└── pyproject.toml                   # Project dependencies
```
