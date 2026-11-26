# Psycholinguistic Reversal Benchmark Results

## External Dataset Test Set (71 passages from `test_ood.jsonl`)

| Method | Diff Score ↓ | Unatt. Bigram ↓ | Trigram PPL ↓ | Mean Dep Len ↓ | Red. ORC /100w ↓ | Stacked N /100w ↓ | Denom. V /100w ↓ | Inanim. Subj /100w ↓ | Cos Sim ↑ | NLI Min ↑ | NLI Contra ↓ | GPT-2 PPL ↓ | Acc Mean ↑ | Acc Min ↑ | Code Pres ↑ | Latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 34.1 | 0.437 | 4478.8 | 2.45 | 0.18 | 0.87 | 0.16 | 0.00 | 1.000 | 0.995 | 0.001 | 17.5 | 0.942 | 0.835 | 1.000 | 0 |
| Classical (spaCy) | 33.3 | 0.436 | 4472.1 | 2.45 | 0.00 | 0.87 | 0.16 | 0.00 | 1.000 | 0.995 | 0.001 | 17.3 | 0.939 | 0.813 | 1.000 | 27 |
| Qwen2.5-3B ZeroShot | 33.7 | 0.444 | 4922.6 | 2.45 | 0.16 | 0.80 | 0.12 | 0.00 | 0.984 | 0.994 | 0.001 | 18.5 | 0.948 | 0.845 | 1.000 | 1550 |
| Qwen2.5-3B Best-of-6 | 32.3 | 0.433 | 4593.5 | 2.45 | 0.13 | 0.66 | 0.11 | 0.00 | 0.984 | 0.988 | 0.014 | 18.1 | 0.947 | 0.845 | 1.000 | 4537 |
| Hybrid (Best-of-6+Classical) | 31.8 | 0.432 | 4580.2 | 2.44 | 0.01 | 0.66 | 0.11 | 0.00 | 0.984 | 0.988 | 0.014 | 18.1 | 0.945 | 0.830 | 1.000 | 4561 |
| Qwen2.5-3B LoRA | 34.2 | 0.449 | 5170.2 | 2.44 | 0.13 | 0.91 | 0.13 | 0.00 | 0.993 | 0.984 | 0.001 | 18.0 | 0.943 | 0.831 | 1.000 | 3044 |
| Hybrid (LoRA+Classical) | 33.7 | 0.448 | 5156.7 | 2.44 | 0.00 | 0.91 | 0.13 | 0.00 | 0.993 | 0.984 | 0.001 | 18.0 | 0.941 | 0.817 | 1.000 | 3067 |

### Per-Domain Breakdown

| Method | coding Diff ↓ | coding NLI ↑ | coding Acc Min ↑ | essay Diff ↓ | essay NLI ↑ | essay Acc Min ↑ | technical_docs Diff ↓ | technical_docs NLI ↑ | technical_docs Acc Min ↑ | workplace_chat Diff ↓ | workplace_chat NLI ↑ | workplace_chat Acc Min ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 29.8 | 0.992 | 0.837 | 34.9 | 0.997 | 0.894 | 34.4 | 0.996 | 0.845 | 36.9 | 0.994 | 0.763 |
| Classical (spaCy) | 29.2 | 0.992 | 0.773 | 34.5 | 0.997 | 0.867 | 34.4 | 0.996 | 0.843 | 34.9 | 0.995 | 0.765 |
| Qwen2.5-3B ZeroShot | 30.3 | 0.989 | 0.856 | 35.9 | 0.998 | 0.892 | 33.0 | 0.992 | 0.822 | 35.3 | 0.997 | 0.809 |
| Qwen2.5-3B Best-of-6 | 29.6 | 0.992 | 0.854 | 35.1 | 0.979 | 0.890 | 32.3 | 0.983 | 0.820 | 32.3 | 0.997 | 0.818 |
| Hybrid (Best-of-6+Classical) | 29.3 | 0.992 | 0.823 | 34.8 | 0.979 | 0.861 | 32.1 | 0.983 | 0.818 | 30.6 | 0.997 | 0.816 |
| Qwen2.5-3B LoRA | 29.8 | 0.992 | 0.843 | 35.8 | 0.960 | 0.890 | 34.7 | 0.990 | 0.834 | 36.5 | 0.996 | 0.758 |
| Hybrid (LoRA+Classical) | 29.2 | 0.992 | 0.788 | 35.6 | 0.960 | 0.889 | 34.6 | 0.990 | 0.833 | 35.0 | 0.996 | 0.758 |

### Out-of-Lexicon vs In-Lexicon Breakdown

| Method | OOL Diff ↓ | OOL NLI ↑ | OOL Acc Mean ↑ | OOL Acc Min ↑ | In-Lex Diff ↓ | In-Lex NLI ↑ | In-Lex Acc Mean ↑ | In-Lex Acc Min ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 34.1 | 0.995 | 0.942 | 0.835 | 0.0 | 0.000 | 0.000 | 0.000 |
| Classical (spaCy) | 33.3 | 0.995 | 0.939 | 0.813 | 0.0 | 0.000 | 0.000 | 0.000 |
| Qwen2.5-3B ZeroShot | 33.7 | 0.994 | 0.948 | 0.845 | 0.0 | 0.000 | 0.000 | 0.000 |
| Qwen2.5-3B Best-of-6 | 32.3 | 0.988 | 0.947 | 0.845 | 0.0 | 0.000 | 0.000 | 0.000 |
| Hybrid (Best-of-6+Classical) | 31.8 | 0.988 | 0.945 | 0.830 | 0.0 | 0.000 | 0.000 | 0.000 |
| Qwen2.5-3B LoRA | 34.2 | 0.984 | 0.943 | 0.831 | 0.0 | 0.000 | 0.000 | 0.000 |
| Hybrid (LoRA+Classical) | 33.7 | 0.984 | 0.941 | 0.817 | 0.0 | 0.000 | 0.000 | 0.000 |

## Blind Pairwise LLM-as-a-Judge (Position-Swapped vs Original)

| Method (vs Original) | Pairs | Win % ↑ | Tie % | Loss % ↓ | Agree % | Clarity ↑ | Naturalness ↑ | Meaning ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical (spaCy) | 35 | 0.0% | 85.7% | 14.3% | 100.0% | 4.80 | 4.79 | 4.93 |
| Qwen2.5-3B ZeroShot | 35 | 0.0% | 74.3% | 25.7% | 88.6% | 4.69 | 4.69 | 4.99 |
| Qwen2.5-3B Best-of-6 | 35 | 0.0% | 65.7% | 34.3% | 94.3% | 4.64 | 4.60 | 4.96 |
| Hybrid (Best-of-6+Classical) | 35 | 0.0% | 57.1% | 42.9% | 94.3% | 4.51 | 4.46 | 4.89 |
| Qwen2.5-3B LoRA | 35 | 0.0% | 74.3% | 25.7% | 85.7% | 4.74 | 4.71 | 4.84 |
| Hybrid (LoRA+Classical) | 35 | 0.0% | 65.7% | 34.3% | 88.6% | 4.63 | 4.59 | 4.79 |
