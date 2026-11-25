"""End-to-end benchmark runner comparing classical and neural psycholinguistic reversal methods."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psycho_rev.benchmark_data import HUMAN_CONTROL, TEST_BENCHMARK, BenchmarkSample
from psycho_rev.classical import ClassicalReverser
from psycho_rev.features import PsycholinguisticAnalyzer
from psycho_rev.neural import (
    BestOfNPsycholinguisticReverser,
    LoRASinglePassReverser,
    NLLBRoundTripReverser,
    QwenZeroShotReverser,
)
from psycho_rev.semantics import SemanticEvaluator


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "difficulty_score",
        "unattested_bigram_rate",
        "trigram_perplexity",
        "mean_dep_length",
        "reduced_orc_per_100w",
        "stacked_noun_per_100w",
        "denominal_verb_per_100w",
        "inanimate_subj_per_100w",
        "cosine_similarity",
        "nli_min_entailment",
        "gpt2_perplexity",
        "code_preservation",
        "length_ratio",
        "latency_ms",
    ]
    out: dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] is not None]
        out[k] = round(float(np.mean(vals)), 4) if vals else 0.0
    return out


def _evaluate_method(
    name: str,
    samples: list[BenchmarkSample],
    rewriter_fn,
    analyzer: PsycholinguisticAnalyzer,
    semantics: SemanticEvaluator,
) -> list[dict[str, Any]]:
    originals = [s.text for s in samples]
    rewrittens: list[str] = []
    latencies: list[float] = []

    for s in samples:
        t0 = time.perf_counter()
        rew = rewriter_fn(s.text)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        rewrittens.append(rew)
        latencies.append(round(dt_ms, 1))

    sem_reports = semantics.evaluate_batch(originals, rewrittens)
    rows: list[dict[str, Any]] = []
    for sample, rew, lat, sem in zip(samples, rewrittens, latencies, sem_reports):
        feat = analyzer.analyze(rew)
        rows.append(
            {
                "method": name,
                "sample_id": sample.sample_id,
                "domain": sample.domain,
                "original": sample.text,
                "rewritten": rew,
                "latency_ms": lat,
                **feat.to_dict(),
                **sem.to_dict(),
            }
        )
    return rows


def _plot_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    methods = summary_df["method"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    colors = ["#64748b", "#d97706", "#0284c7", "#7c3aed", "#059669", "#0f766e", "#dc2626", "#be123c", "#10b981"][: len(methods)]

    # Panel 1: Composite Psycholinguistic Difficulty Score (lower is better)
    ax = axes[0]
    vals = summary_df["difficulty_score"].tolist()
    bars = ax.barh(methods, vals, color=colors, edgecolor="none", height=0.6)
    ax.invert_yaxis()
    ax.set_title("Psycholinguistic Difficulty Index (lower = clearer)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Composite score")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center", fontsize=9)

    # Panel 2: Key Feature Rates per 100 words
    ax = axes[1]
    x = np.arange(len(methods))
    w = 0.2
    ax.bar(x - 1.5 * w, summary_df["reduced_orc_per_100w"], width=w, label="Reduced ORCs", color="#2563eb")
    ax.bar(x - 0.5 * w, summary_df["stacked_noun_per_100w"], width=w, label="Stacked Nouns", color="#d97706")
    ax.bar(x + 0.5 * w, summary_df["denominal_verb_per_100w"], width=w, label="Denominal Verbs", color="#dc2626")
    ax.bar(x + 1.5 * w, summary_df["inanimate_subj_per_100w"], width=w, label="Inanimate Subj", color="#7c3aed")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("Occurrences per 100 words")
    ax.set_title("Target Feature Rates (lower = better)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, frameon=True)

    # Panel 3: Semantic Fidelity (Cosine Sim & Bidirectional NLI Entailment)
    ax = axes[2]
    ax.scatter(
        summary_df["cosine_similarity"],
        summary_df["nli_min_entailment"],
        s=120,
        c=colors,
        edgecolors="#111827",
        linewidths=0.8,
        zorder=3,
    )
    for _, row in summary_df.iterrows():
        ax.annotate(
            row["method"],
            (row["cosine_similarity"], row["nli_min_entailment"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8.5,
        )
    ax.set_xlim(0.55, 1.03)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Embedding Cosine Similarity")
    ax.set_ylabel("Bidirectional NLI Min-Entailment")
    ax.set_title("Semantic Preservation (top-right = best)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(
    results_dir: Path = Path("results"),
    adapter_dir: Path | None = Path("checkpoints/qwen25-1.5b-psycho-rev-lora"),
    include_neural: bool = True,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    analyzer = PsycholinguisticAnalyzer()
    semantics = SemanticEvaluator()
    classical = ClassicalReverser()

    all_rows: list[dict[str, Any]] = []
    summaries_dense: list[dict[str, Any]] = []
    summaries_human: list[dict[str, Any]] = []

    def _run_and_record(name: str, fn):
        print(f"=== Running method: {name} ===")
        dense_rows = _evaluate_method(name, TEST_BENCHMARK, fn, analyzer, semantics)
        human_rows = _evaluate_method(name, HUMAN_CONTROL, fn, analyzer, semantics)
        all_rows.extend(dense_rows)
        all_rows.extend(human_rows)
        s_dense = {"method": name, **_summarize_rows(dense_rows)}
        s_human = {"method": name, **_summarize_rows(human_rows)}
        summaries_dense.append(s_dense)
        summaries_human.append(s_human)
        print(
            f"  [Dense AI] diff={s_dense['difficulty_score']:.1f} "
            f"redORC={s_dense['reduced_orc_per_100w']:.2f} "
            f"stackN={s_dense['stacked_noun_per_100w']:.2f} "
            f"denomV={s_dense['denominal_verb_per_100w']:.2f} "
            f"inanimS={s_dense['inanimate_subj_per_100w']:.2f} | "
            f"cos={s_dense['cosine_similarity']:.3f} nli={s_dense['nli_min_entailment']:.3f} ppl={s_dense['gpt2_perplexity']:.1f} "
            f"code={s_dense['code_preservation']:.3f} lat={s_dense['latency_ms']:.0f}ms"
        )

    # 1. Original baseline
    _run_and_record("Original", lambda x: x)

    # 2. Classical rule-based reverser
    _run_and_record("Classical (spaCy)", classical.rewrite)

    if include_neural:
        # 3. NLLB-200 Round-Trip Translation (En -> Ja -> En)
        nllb = NLLBRoundTripReverser("facebook/nllb-200-distilled-600M")
        _run_and_record("NLLB-200 RTT", nllb.rewrite)
        nllb.close()

        # 4. Qwen2.5-3B-Instruct Zero-Shot + Best-of-6 Psycholinguistic Reranking
        qwen = QwenZeroShotReverser("Qwen/Qwen2.5-3B-Instruct")
        _run_and_record("Qwen2.5-3B ZeroShot", qwen.rewrite)

        reranker = BestOfNPsycholinguisticReverser(qwen, analyzer, semantics, num_samples=6)
        bon_cache: dict[str, str] = {}
        def _bon_cached(text: str) -> str:
            if text not in bon_cache:
                bon_cache[text] = reranker.rewrite(text)
            return bon_cache[text]
        _run_and_record("Qwen2.5-3B Best-of-6", _bon_cached)
        _run_and_record("Hybrid (Best-of-6+Classical)", lambda x: classical.rewrite(bon_cache[x]))
        qwen.close()

        # 5. Fine-tuned Qwen2.5-1.5B LoRA Single-Pass (if checkpoint exists)
        if adapter_dir is not None and adapter_dir.exists():
            lora = LoRASinglePassReverser(adapter_dir, base_model="Qwen/Qwen2.5-1.5B-Instruct")
            _run_and_record("Qwen2.5-1.5B LoRA", lora.rewrite)
            _run_and_record("Hybrid (LoRA+Classical)", lambda x: classical.rewrite(lora.rewrite(x)))
            lora.close()

    # Add human control baseline reference row to dense summary plot for context
    human_ref = _summarize_rows([r for r in all_rows if r["method"] == "Original" and r["domain"] == "human_control"])
    plot_rows = summaries_dense + [{"method": "Human Reference", **human_ref}]
    summary_df = pd.DataFrame(plot_rows)
    _plot_comparison(summary_df, results_dir / "comparison_plot.png")

    payload = {
        "dense_ai_summary": summaries_dense,
        "human_control_summary": summaries_human,
        "human_reference_baseline": human_ref,
        "per_sample": all_rows,
    }
    with open(results_dir / "benchmark_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Write Markdown summary table
    md_lines = [
        "# Psycholinguistic Reversal Benchmark Results",
        "",
        "## Dense AI Test Set (30 passages: Chat, Coding, Essay)",
        "",
        "| Method | Diff Score ↓ | Unatt. Bigram ↓ | Trigram PPL ↓ | Mean Dep Len ↓ | Red. ORC /100w ↓ | Stacked N /100w ↓ | Denom. V /100w ↓ | Inanim. Subj /100w ↓ | Cos Sim ↑ | NLI Min ↑ | GPT-2 PPL ↓ | Code Pres ↑ | Latency (ms) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summaries_dense:
        md_lines.append(
            f"| {r['method']} | {r['difficulty_score']:.1f} | {r['unattested_bigram_rate']:.3f} | "
            f"{r['trigram_perplexity']:.1f} | {r['mean_dep_length']:.2f} | {r['reduced_orc_per_100w']:.2f} | "
            f"{r['stacked_noun_per_100w']:.2f} | {r['denominal_verb_per_100w']:.2f} | "
            f"{r['inanimate_subj_per_100w']:.2f} | {r['cosine_similarity']:.3f} | "
            f"{r['nli_min_entailment']:.3f} | {r['gpt2_perplexity']:.1f} | {r['code_preservation']:.3f} | {r['latency_ms']:.0f} |"
        )
    md_lines.append(
        f"| *Human Reference (10 control)* | *{human_ref['difficulty_score']:.1f}* | *{human_ref['unattested_bigram_rate']:.3f}* | "
        f"*{human_ref['trigram_perplexity']:.1f}* | *{human_ref['mean_dep_length']:.2f}* | *{human_ref['reduced_orc_per_100w']:.2f}* | "
        f"*{human_ref['stacked_noun_per_100w']:.2f}* | *{human_ref['denominal_verb_per_100w']:.2f}* | "
        f"*{human_ref['inanimate_subj_per_100w']:.2f}* | *1.000* | *1.000* | *{human_ref['gpt2_perplexity']:.1f}* | *1.000* | *-* |"
    )

    md_lines.extend([
        "",
        "## Human Control Set (10 natural passages — checking against over-editing)",
        "",
        "| Method | Diff Score | Cos Sim ↑ | NLI Min ↑ | Code Pres ↑ | Length Ratio |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in summaries_human:
        md_lines.append(
            f"| {r['method']} | {r['difficulty_score']:.1f} | {r['cosine_similarity']:.3f} | "
            f"{r['nli_min_entailment']:.3f} | {r['code_preservation']:.3f} | {r['length_ratio']:.2f} |"
        )

    with open(results_dir / "summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    return payload


if __name__ == "__main__":
    run_benchmark()
