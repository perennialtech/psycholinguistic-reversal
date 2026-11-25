# Psycholinguistic Reversal Benchmark Results

## Dense AI Test Set (30 passages: Chat, Coding, Essay)

| Method | Diff Score ↓ | Unatt. Bigram ↓ | Trigram PPL ↓ | Mean Dep Len ↓ | Red. ORC /100w ↓ | Stacked N /100w ↓ | Denom. V /100w ↓ | Inanim. Subj /100w ↓ | Cos Sim ↑ | NLI Min ↑ | GPT-2 PPL ↓ | Code Pres ↑ | Latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 129.1 | 0.752 | 81383.1 | 2.45 | 1.51 | 6.99 | 4.98 | 2.99 | 1.000 | 0.932 | 847.8 | 1.000 | 0 |
| Classical (spaCy) | 52.6 | 0.686 | 50574.1 | 2.49 | 0.15 | 1.83 | 0.51 | 0.29 | 0.955 | 0.989 | 798.1 | 1.000 | 6 |
| NLLB-200 RTT | 61.3 | 0.510 | 15422.9 | 2.44 | 0.43 | 3.37 | 0.77 | 1.13 | 0.803 | 0.597 | 302.4 | 0.967 | 708 |
| Qwen2.5-3B ZeroShot | 66.0 | 0.609 | 30522.2 | 2.42 | 0.93 | 3.87 | 1.15 | 0.28 | 0.863 | 0.687 | 249.3 | 0.967 | 793 |
| Qwen2.5-3B Best-of-6 | 50.6 | 0.543 | 17660.3 | 2.32 | 0.60 | 2.63 | 0.49 | 0.12 | 0.853 | 0.888 | 205.5 | 0.967 | 1872 |
| Hybrid (Best-of-6+Classical) | 36.2 | 0.524 | 15331.4 | 2.22 | 0.17 | 0.98 | 0.12 | 0.00 | 0.851 | 0.889 | 215.4 | 0.967 | 6 |
| Qwen2.5-1.5B LoRA | 44.1 | 0.455 | 12808.7 | 2.24 | 0.49 | 1.88 | 0.36 | 0.54 | 0.737 | 0.503 | 133.8 | 1.000 | 1706 |
| Hybrid (LoRA+Classical) | 30.1 | 0.447 | 11878.7 | 2.19 | 0.00 | 0.48 | 0.09 | 0.07 | 0.735 | 0.567 | 146.2 | 1.000 | 1706 |
| *Human Reference (10 control)* | *31.1* | *0.428* | *7190.8* | *2.53* | *0.00* | *0.00* | *0.00* | *0.46* | *1.000* | *1.000* | *145.3* | *1.000* | *-* |

## Human Control Set (10 natural passages — checking against over-editing)

| Method | Diff Score | Cos Sim ↑ | NLI Min ↑ | Code Pres ↑ | Length Ratio |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Original | 31.1 | 1.000 | 0.972 | 1.000 | 1.00 |
| Classical (spaCy) | 31.1 | 1.000 | 0.972 | 1.000 | 1.00 |
| NLLB-200 RTT | 26.4 | 0.921 | 0.574 | 1.000 | 1.03 |
| Qwen2.5-3B ZeroShot | 31.8 | 0.968 | 0.859 | 1.000 | 1.04 |
| Qwen2.5-3B Best-of-6 | 27.1 | 0.964 | 0.956 | 1.000 | 1.06 |
| Hybrid (Best-of-6+Classical) | 27.1 | 0.964 | 0.956 | 1.000 | 1.06 |
| Qwen2.5-1.5B LoRA | 29.6 | 0.897 | 0.897 | 1.000 | 1.07 |
| Hybrid (LoRA+Classical) | 27.1 | 0.893 | 0.883 | 1.000 | 1.08 |
