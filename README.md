# psycholinguistic-reversal

Empirical study and runnable inference toolkit for **classically and neurally reversing** the six psycholinguistic difficulty features identified in [*Why is AI-generated language so hard to understand?* (Clark, Martin, & Mahowald, 2026; arXiv:2610.29162v1)](https://arxiv.org/abs/2610.29162).

## Overview

Modern post-trained LLMs overuse constructions that compress information density at the expense of human comprehension:

1. **Low-frequency / unattested n-grams** (measured against NLTK Brown + Reuters + Webtext plus modern technical vocabulary)
2. **Long syntactic dependencies** (mean and sentence-length-standardized dependency arc length)
3. **Reduced object relative clauses (ORCs)** (omitting optional `that` / `which`)
4. **Stacked noun compounds** (3+ consecutive nouns in a modifier chain, e.g., *reader-patience budget*)
5. **Denominal verbs** (ad-hoc noun-to-verb conversions, e.g., *footnote it*, *calendar it*, *gate*, *sunset*)
6. **Inanimate subjects with animate verbs** (selectional-restriction violations, e.g., *`is_file()` swallows that*, *this asymmetry bites*)

This repository implements detectors for all six features, calibrated NLI/fluency/code-preservation evaluators, markdown-structure-preserving rewriting, and seven reversal pipelines spanning deterministic spaCy rules, round-trip translation, zero-shot and Best-of-N neural rewriting, QLoRA SFT + SimPO distillation, and hybrid neural+classical post-processing.

![Benchmark comparison](results/comparison_plot.png)

## Benchmark Results

Evaluated across both **single-sentence** (30 dense AI passages + 10 human control) and **multi-paragraph** (18 markdown/multi-paragraph dense AI passages + 6 human control) splits spanning **chat**, **coding**, and **essay** domains, using `cross-encoder/nli-deberta-v3-base` with paragraph-chunked NLI and strided GPT-2 perplexity on an RTX 3080 (10 GB VRAM).

### Single-Sentence Dense AI Test Set (30 passages)

| Method | Diff Score ↓ | Unatt. Bigram ↓ | Trigram PPL ↓ | Mean Dep Len ↓ | Red. ORC /100w ↓ | Stacked N /100w ↓ | Denom. V /100w ↓ | Inanim. Subj /100w ↓ | Cos Sim ↑ | NLI Min ↑ | NLI Contra ↓ | GPT-2 PPL ↓ | Code Pres ↑ | Latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 129.5 | 0.752 | 57501.1 | 2.45 | 1.51 | 6.98 | 5.13 | 2.99 | 1.000 | 0.986 | 0.002 | 847.8 | 1.000 | 0 |
| Classical (spaCy) | 53.4 | 0.674 | 33893.0 | 2.48 | 0.15 | 1.69 | 0.91 | 0.28 | 0.954 | 0.963 | 0.033 | 741.7 | 1.000 | 6 |
| NLLB-200 RTT (En→Ja→En) | 60.5 | 0.510 | 13295.0 | 2.44 | 0.43 | 3.23 | 0.77 | 1.13 | 0.803 | 0.698 | 0.100 | 302.4 | 0.967 | 713 |
| Qwen2.5-3B ZeroShot | 66.5 | 0.609 | 21127.7 | 2.42 | 0.93 | 4.03 | 1.15 | 0.28 | 0.863 | 0.824 | 0.033 | 249.3 | 0.967 | 789 |
| Qwen2.5-3B Best-of-6 | 54.7 | 0.546 | 12057.3 | 2.37 | 0.68 | 3.14 | 0.51 | 0.27 | 0.863 | 0.957 | 0.033 | 195.1 | 1.000 | 1895 |
| **Hybrid (Best-of-6 + Classical)** | **37.1** | **0.520** | **10127.9** | **2.25** | **0.17** | **1.02** | **0.24** | **0.00** | **0.859** | **0.957** | **0.033** | **207.0** | **1.000** | **1900** |
| Qwen2.5-3B LoRA (SFT+SimPO) | 46.8 | 0.513 | 9928.6 | 2.27 | 1.20 | 1.99 | 0.17 | 0.24 | 0.811 | 0.764 | 0.001 | 156.6 | 1.000 | 1921 |
| Hybrid (LoRA + Classical) | 32.4 | 0.482 | 8116.7 | 2.28 | 0.15 | 0.37 | 0.00 | 0.23 | 0.807 | 0.764 | 0.001 | 152.9 | 1.000 | 1927 |
| *Human Reference (10 control)* | *31.1* | *0.428* | *6804.1* | *2.53* | *0.00* | *0.00* | *0.00* | *0.46* | *1.000* | *1.000* | *0.000* | *145.3* | *1.000* | *-* |

### Multi-Paragraph Dense AI Test Set (18 passages)

| Method | Diff Score ↓ | Unatt. Bigram ↓ | Trigram PPL ↓ | Mean Dep Len ↓ | Red. ORC /100w ↓ | Stacked N /100w ↓ | Denom. V /100w ↓ | Inanim. Subj /100w ↓ | Cos Sim ↑ | NLI Min ↑ | NLI Contra ↓ | GPT-2 PPL ↓ | Code Pres ↑ | Latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 89.0 | 0.762 | 56773.9 | 2.80 | 0.18 | 2.99 | 3.63 | 1.43 | 1.000 | 0.995 | 0.000 | 299.6 | 1.000 | 0 |
| Classical (spaCy) | 50.7 | 0.724 | 41150.6 | 2.71 | 0.06 | 0.84 | 0.66 | 0.25 | 0.977 | 0.991 | 0.000 | 298.1 | 1.000 | 22 |
| NLLB-200 RTT | 54.4 | 0.555 | 11188.8 | 3.11 | 0.18 | 1.98 | 0.42 | 0.62 | 0.807 | 0.745 | 0.057 | 176.4 | 0.949 | 3074 |
| Qwen2.5-3B ZeroShot | 45.3 | 0.583 | 14362.4 | 2.42 | 0.30 | 1.52 | 0.44 | 0.11 | 0.920 | 0.993 | 0.000 | 118.8 | 0.957 | 3132 |
| Qwen2.5-3B Best-of-6 | 38.3 | 0.539 | 9557.9 | 2.45 | 0.14 | 1.08 | 0.10 | 0.00 | 0.902 | 0.993 | 0.000 | 109.2 | 1.000 | 7865 |
| **Hybrid (Best-of-6 + Classical)** | **32.3** | **0.532** | **9227.5** | **2.41** | **0.00** | **0.14** | **0.06** | **0.00** | **0.901** | **0.992** | **0.000** | **116.7** | **1.000** | **7887** |
| Qwen2.5-3B LoRA (SFT+SimPO) | 33.0 | 0.472 | 5508.6 | 2.31 | 0.13 | 0.70 | 0.08 | 0.05 | 0.859 | 0.906 | 0.000 | 75.8 | 1.000 | 8319 |
| Hybrid (LoRA + Classical) | 29.1 | 0.466 | 5360.2 | 2.30 | 0.00 | 0.09 | 0.08 | 0.05 | 0.858 | 0.889 | 0.000 | 79.9 | 1.000 | 8340 |
| *Human Reference Multi-Para (6 control)* | *34.0* | *0.497* | *6739.6* | *2.59* | *0.00* | *0.56* | *0.00* | *0.00* | *1.000* | *1.000* | *0.000* | *49.4* | *1.000* | *-* |

### Key Findings

* **Hybrid (`Best-of-6 + Classical`) is the highest-fidelity configuration**: Achieves `100%` code preservation across both single-sentence and multi-paragraph splits, drops difficulty to `37.1` (single) and `32.3` (multi-para), and maintains `0.957–0.992` NLI entailment while leaving human control prose untouched (`0.997` NLI).
* **Scaled `Qwen2.5-3B` QLoRA (SFT + SimPO) resolves semantic collapse**: Trained on 131 verified Best-of-8 pairs + 30 identity regularizers, the 3B QLoRA adapter reaches `0.906` NLI (`0.000` contradiction) and `75.8` GPT-2 PPL on multi-paragraph text and `0.893` NLI on single-sentence chat—a major improvement over the earlier 1.5B POC (`0.503` overall, `0.208` chat).
* **NP-boundary-aware classical rules are safe as a deterministic post-pass**: Restricting noun-stacking to single NP subtrees and gating denominal frames on context eliminates relative-clause corruption while preserving markdown structure in 6–22 ms.

## Installation

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(p) for p in ('wordnet','omw-1.4','brown','reuters','webtext','punkt','punkt_tab')]"
```

## Usage

### Analyze psycholinguistic features

```bash
psycho-rev analyze "Your instinct is right; the convoluted sentence you are currently reading is quietly eating its own reader-patience budget — footnote it."
psycho-rev analyze --file draft.md
```

### Rewrite text (classical, neural, or hybrid)

```bash
# Fast symbolic/spaCy reverser (CPU, ~6 ms)
psycho-rev rewrite --method classical "Right now \`is_file()\` swallows that permission error, which quietly sabotages the cache invalidation pipeline you added in \`src/cache/store.py\`."

# Neural and hybrid options (GPU recommended)
psycho-rev rewrite --method best-of-n --samples 6 --file draft.md --plain
psycho-rev rewrite --method hybrid-bon --file draft.md --plain
psycho-rev rewrite --method lora --adapter-dir checkpoints/qwen25-3b-psycho-rev-lora "..."
psycho-rev rewrite --method hybrid-lora --adapter-dir checkpoints/qwen25-3b-psycho-rev-lora "..."
```

### Train QLoRA/LoRA (SFT + SimPO) and run full benchmark

```bash
python -m psycho_rev.train_lora --resume --epochs 3 --dpo-epochs 2
python -m psycho_rev.evaluate --adapter-dir checkpoints/qwen25-3b-psycho-rev-lora
pytest -v
```

## Project Layout

* `src/psycho_rev/features.py` — Detectors for all six psycholinguistic features (`ngram_ref_v2.pkl` + NP-boundary-safe `find_stacked_noun_spans`) and composite Difficulty Index
* `src/psycho_rev/semantics.py` — MiniLM cosine similarity, dynamic-label DeBERTa-v3 chunked bidirectional NLI + contradiction, and strided GPT-2 fluency PPL
* `src/psycho_rev/code_guard.py` — Placeholder masking/restoration (fenced blocks, inline code, URLs, dotted identifiers, filenames) and markdown-structure-preserving wrapper
* `src/psycho_rev/classical.py` — Deterministic spaCy dependency-tree + context-gated lexical frame reverser
* `src/psycho_rev/neural.py` — NLLB-200 RTT, Qwen2.5 zero-shot, Best-of-N reranker with hard code preservation, auto-base-detecting `LoRASinglePassReverser`, and `HybridReverser`
* `src/psycho_rev/train_lora.py` — Resumable Best-of-8 teacher distillation + QLoRA SFT and reference-free SimPO preference optimization
* `src/psycho_rev/evaluate.py` — Single-sentence and multi-paragraph benchmark runner, Markdown summary, and 3-panel visualization
* `results/` — Saved benchmark metrics (`benchmark_results.json`, `summary.md`, `distilled_train_pairs.json`, `comparison_plot.png`)
