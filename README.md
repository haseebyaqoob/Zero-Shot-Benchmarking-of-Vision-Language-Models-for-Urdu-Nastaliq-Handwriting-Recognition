# Benchmarking Vision-Language Models for Urdu Nastaliq Handwriting Recognition

Code, per-image predictions, and logs for the paper *Benchmarking Vision-Language Models for Urdu Nastaliq Handwriting Recognition*.

**Authors:** Haseeb Yaqoob, Fatima Azfar, Emaan Arif Khan, Usman Amjad
Department of Computer Science, NED University of Engineering & Technology, Karachi, Pakistan

📄 **Paper:** [`paper/benchmarking_vlms_urdu_nastaliq_htr.pdf`](paper/benchmarking_vlms_urdu_nastaliq_htr.pdf)

## Overview

We benchmark six systems on the Urdu Nastaliq Handwritten Dataset (UNHD) with no fine-tuning: Tesseract, UTRNet-Large (trained on printed Urdu), two open-source VLMs (Qwen2.5-VL-7B, Qwen3-VL-8B), and two commercial VLMs (Gemini 3.5 Flash-Lite, Claude Sonnet 5).

- **Part A:** direct transcription of 2,465 single-line images.
- **Part B:** can a VLM correct a separate OCR system's output, given the image plus the noisy text?

We did not compare against handwriting-trained recognizers, which would likely do better in-domain. This is a zero-shot study, not a state-of-the-art claim.

## Results

### Part A (diacritic-sensitive, N = 2,465, lower is better)

| System | CER | WER |
|---|---|---|
| **Gemini 3.5 Flash-Lite** | **0.1718** | **0.3772** |
| Claude Sonnet 5 ¹ | 0.2656 | 0.5225 |
| UTRNet-Large | 0.3084 | 0.7827 |
| Qwen3-VL-8B | 0.4861 | 0.8380 |
| Qwen2.5-VL-7B | 0.6464 | 0.9495 |
| Tesseract | 0.7215 | 1.1793 |

¹ Excludes 5 images (0.20%) the model declined to transcribe as illegible. Scored as maximal error, CER is 0.2671; the ranking is unchanged.

MiniCPM-V-4.5 and InternVL3.5-8B were excluded from the ranking. InternVL3.5 produced fluent but unrelated Urdu on the full set (CER 1.0633); MiniCPM-V-4.5 was too unstable to evaluate in full (paper §V-C).

### Part B: correction gain

CER reduction (%) relative to the input text, and relative to the corrector's own Part A CER (which isolates whether the supplied text helped):

| Corrector | Input | vs. input | vs. own Part A |
|---|---|---|---|
| Gemini 3.5 Flash-Lite | Tesseract | 77.6 | 6.1 |
| Gemini 3.5 Flash-Lite | UTRNet | 54.3 | 17.9 |
| Claude Sonnet 5 | Tesseract | 64.3 | 3.1 |
| Claude Sonnet 5 | UTRNet | 32.6 | 21.9 |
| Qwen3-VL-8B | Tesseract | 33.1 | 0.7 (CI includes 0) |
| Qwen3-VL-8B | UTRNet | 3.9 | 39.0 ² |
| Qwen2.5-VL-7B | Tesseract | 17.1 | 7.4 |
| Qwen2.5-VL-7B | UTRNet | -9.0 | 48.0 ² |

² Largely mechanical: UTRNet's text is far better than these models' own reading, so adopting it looks like a gain whether or not correction adds value.

**Takeaways:** reduction against a weak baseline greatly overstates the benefit of supplying its text; gains that are not mechanical artifacts appear only for the two commercial correctors given UTRNet's more legible text; and one weaker corrector (Qwen2.5-VL-7B) made UTRNet's text worse on average. Confidence intervals and the legibility analysis are in the paper.

## Repository structure

```
.
├── paper/                  # PDF and LaTeX source
├── notebooks/              # one notebook per system (Part A / Part B)
├── results/
│   ├── part_a/             # per-image predictions (CSV) + summaries
│   └── part_b/
├── docs/                   # research log, negative results, Sonnet 5 reports
└── data/README.md          # how to get UNHD (dataset not included)
```

Prediction CSVs include `file_id`, `writer_id`, `ground_truth`, `prediction`, and CER/WER columns, so the exact images can be identified.

## Reproducing

1. Download UNHD (see `data/README.md`). The dataset is **not** redistributed here.
2. Open models ran under vLLM on a single Tesla T4. API models need your own keys (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`). Never commit keys.
3. All VLMs use the same fixed prompt (paper §IV-B). Open models use greedy decoding. Claude Sonnet 5's decoding cannot be configured, so small run-to-run differences are expected.

**Evaluation set:** 2,465 image/ground-truth pairs, 837 sentence templates, 336 writers, built from 7,319 valid UNHD pairs after de-duplication. Twelve sentences with fewer than two available writers were excluded.

## Caveats

One English prompt, one run per system, one version of each commercial model; open-source models ran 4-bit quantized; UNHD may appear in VLM pretraining data; six images have mislabeled ground truth (kept in all scores). See paper §VII.

## Citation

TODO: add the BibTeX entry once the paper is published.

## License

TODO: choose a license for the code. UNHD has its own license and is not covered by this repo's.
