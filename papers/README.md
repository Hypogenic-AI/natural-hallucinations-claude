# Downloaded Papers

This directory contains papers relevant to studying natural hallucinations in LLMs.

## Paper List

### Core Papers (Highly Relevant)

1. **TruthfulQA: Measuring How Models Mimic Human Falsehoods** (2021)
   - File: `2109.07958_TruthfulQA.pdf`
   - Authors: Lin, Hilton, Evans
   - Citations: 2693
   - Key: Benchmark for imitative falsehoods, inverse scaling phenomenon

2. **How Language Model Hallucinations Can Snowball** (2023)
   - File: `2305.13534_hallucination_snowball.pdf`
   - Authors: Zhang, Press, Merrill, Liu, Smith
   - Key: LLMs recognize 67-87% of own errors when asked separately

3. **INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection** (2024)
   - File: `2402.03744_INSIDE.pdf`
   - Authors: Chen et al.
   - Venue: ICLR 2024
   - Key: EigenScore for internal state-based detection

4. **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection** (2023)
   - File: `2303.08896_selfcheckgpt.pdf`
   - Authors: Manakul, Liusie, Gales
   - Citations: 685
   - Key: Self-consistency based detection

### Survey Papers

5. **A Survey on Hallucination in Large Language Models** (2023)
   - File: `2311.05232_hallucination_survey_huang.pdf`
   - Citations: 1960
   - Scope: Comprehensive taxonomy

6. **Siren's Song in the AI Ocean** (2023)
   - File: `2309.01219_hallucination_siren_song.pdf`
   - Citations: 819
   - Scope: Detection and mitigation

7. **A Survey on Evaluation of Large Language Models** (2023)
   - File: `2307.03109_llm_eval_survey_comprehensive.pdf`
   - Citations: 2786
   - Scope: Broad evaluation methods

### Detection Methods

8-20. Additional detection papers covering:
- Semantic entropy probes
- Internal states analysis
- Cross-examination methods
- Cost-effective detection
- Zero-resource approaches

### Benchmarks

21-30. Benchmark papers including:
- FactBench, FactKB, HalluLens
- RAGTruth, SimpleQA, ReFACT
- Domain-specific (legal, medical)

### Mitigation Methods

31-35. Papers on:
- TruthX (editing approach)
- R-Tuning (saying "I don't know")
- SaySelf (confidence expression)
- SLED (self logits evolution)

## Total: 35 papers

## Notes

- Papers are named with format: `{arxiv_id}_{short_name}.pdf`
- Chunked versions for reading are in `pages/` subdirectory
- Search results saved in `search_results_*.txt` files
