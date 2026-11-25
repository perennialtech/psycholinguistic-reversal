"""End-to-end benchmark runner comparing classical and neural psycholinguistic reversal methods."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psycho_rev.benchmark_data import (
    HUMAN_CONTROL,
    HUMAN_CONTROL_MULTIPARA,
    MULTIPARA_BENCHMARK,
    TEST_BENCHMARK,
    BenchmarkSample,
)
from psycho_rev.classical import ClassicalReverser
from psycho_rev.features import PsycholinguisticAnalyzer
from psycho_rev.neural import (
    BestOfNPsycholinguisticReverser,
    LoRASinglePassReverser,
    NLLBRoundTripReverser,
    QwenZeroShotReverser,
)
from psycho_rev.semantics import SemanticEvaluator

DOMAINS: tuple[str, ...] = ("chat", "coding", "essay")


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
        "nli_max_contradiction",
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


def _summarize_by_domain(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for dom in DOMAINS:
        dom_rows = [r for r in rows if r.get("domain") == dom]
        if dom_rows:
            out[dom] = _summarize_rows(dom_rows)
    return out


def _evaluate_method(
    name: str,
    samples: list[BenchmarkSample],
    rewriter_fn: Callable[[str], str],
    analyzer: PsycholinguisticAnalyzer,
    semantics: SemanticEvaluator,
    split: str = "single",
    extra_latency_fn: Callable[[str], float] | None = None,
) -> list[dict[str, Any]]:
    originals = [s.text for s in samples]
    rewrittens: list[str] = []
    latencies: list[float] = []

    for s in samples:
        t0 = time.perf_counter()
        rew = rewriter_fn(s.text)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if extra_latency_fn is not None:
            dt_ms += extra_latency_fn(s.text)
        rewrittens.append(rew)
        latencies.append(round(dt_ms, 1))

    sem_reports = semantics.evaluate_batch(originals, rewrittens)
    rows: list[dict[str, Any]] = []
    for sample, rew, lat, sem in zip(samples, rewrittens, latencies, sem_reports):
        feat = analyzer.analyze(rew)
        sem_dict = sem.to_dict()
        sem_dict.setdefault("nli_max_contradiction", 0.0)
        rows.append(
            {
                "method": name,
                "split": split,
                "sample_id": sample.sample_id,
                "domain": sample.domain,
                "original": sample.text,
                "rewritten": rew,
                "latency_ms": lat,
                **feat.to_dict(),
                **sem_dict,
            }
        )
    return rows


def _plot_comparison(
    summaries_dense: list[dict[str, Any]],
    domain_breakdowns: list[dict[str, Any]],
    summaries_multipara: list[dict[str, Any]],
    human_ref: dict[str, float],
    out_path: Path,
) -> None:
    dense_df = pd.DataFrame(summaries_dense)
    methods = dense_df["method"].tolist()
    palette = [
        "#64748b",
        "#d97706",
        "#0284c7",
        "#7c3aed",
        "#059669",
        "#0f766e",
        "#dc2626",
        "#be123c",
        "#2563eb",
    ]
    colors = [palette[i % len(palette)] for i in range(len(methods))]

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), dpi=180)
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Pareto Frontier — Psycholinguistic Difficulty (lower) vs Bidirectional NLI (higher)
    ax = axes[0]
    ax.scatter(
        dense_df["difficulty_score"],
        dense_df["nli_min_entailment"],
        s=130,
        c=colors,
        edgecolors="#111827",
        linewidths=0.9,
        zorder=4,
    )
    ax.scatter(
        [human_ref["difficulty_score"]],
        [1.0],
        s=220,
        marker="*",
        c="#eab308",
        edgecolors="#111827",
        linewidths=0.9,
        zorder=5,
        label="Human Reference",
    )
    for idx, row in dense_df.iterrows():
        dy = 6 if idx % 2 == 0 else -12
        ax.annotate(
            row["method"],
            (row["difficulty_score"], row["nli_min_entailment"]),
            textcoords="offset points",
            xytext=(6, dy),
            fontsize=8,
            color="#1f2937",
        )
    ax.annotate(
        "Human Reference",
        (human_ref["difficulty_score"], 1.0),
        textcoords="offset points",
        xytext=(8, -4),
        fontsize=8,
        fontweight="bold",
        color="#854d0e",
    )
    ax.invert_xaxis()
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Psycholinguistic Difficulty Score (right = clearer)", fontsize=9.5)
    ax.set_ylabel("Bidirectional NLI Min-Entailment (higher = faithful)", fontsize=9.5)
    ax.set_title("Pareto: Difficulty vs Semantic Fidelity", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel 2: Per-Domain Bidirectional NLI Breakdown
    ax = axes[1]
    dom_map = {d["method"]: d for d in domain_breakdowns}
    x = np.arange(len(methods))
    w = 0.25
    dom_colors = {"chat": "#2563eb", "coding": "#059669", "essay": "#d97706"}
    for offset, dom in zip((-w, 0.0, w), DOMAINS):
        vals = [dom_map.get(m, {}).get(dom, {}).get("nli_min_entailment", 0.0) for m in methods]
        ax.bar(x + offset, vals, width=w, label=dom.capitalize(), color=dom_colors[dom], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=28, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Bidirectional NLI Min-Entailment", fontsize=9.5)
    ax.set_title("Domain NLI Fidelity Breakdown", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.5, frameon=True, loc="lower right")

    # Panel 3: Single-Sentence vs Multi-Paragraph Difficulty & NLI Comparison
    ax = axes[2]
    if summaries_multipara:
        mp_map = {r["method"]: r for r in summaries_multipara}
        w3 = 0.36
        single_diff = [r["difficulty_score"] for r in summaries_dense]
        multi_diff = [mp_map.get(m, {}).get("difficulty_score", 0.0) for m in methods]
        single_nli = [r["nli_min_entailment"] for r in summaries_dense]
        multi_nli = [mp_map.get(m, {}).get("nli_min_entailment", 0.0) for m in methods]

        b1 = ax.bar(x - w3 / 2, single_diff, width=w3, label="Single-Sent Diff ↓", color="#0284c7", alpha=0.9)
        b2 = ax.bar(x + w3 / 2, multi_diff, width=w3, label="Multi-Para Diff ↓", color="#7c3aed", alpha=0.9)

        ax2 = ax.twinx()
        ax2.plot(
            x - w3 / 2,
            single_nli,
            color="#0369a1",
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=5,
            label="Single-Sent NLI ↑",
        )
        ax2.plot(
            x + w3 / 2,
            multi_nli,
            color="#5b21b6",
            marker="s",
            linestyle="--",
            linewidth=1.5,
            markersize=5,
            label="Multi-Para NLI ↑",
        )
        ax2.set_ylim(0.0, 1.08)
        ax2.set_ylabel("Bidirectional NLI Min-Entailment", fontsize=9.5)

        for bar, v in zip(b1, single_diff):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(b2, multi_diff):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=28, ha="right", fontsize=8)
        ax.set_ylabel("Psycholinguistic Difficulty Score", fontsize=9.5)
        ax.set_title("Single-Sentence vs Multi-Paragraph", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=True, loc="upper right")
    else:
        vals = dense_df["difficulty_score"].tolist()
        contras = dense_df["nli_max_contradiction"].tolist()
        bars = ax.barh(methods, vals, color=colors, edgecolor="none", height=0.6)
        ax.invert_yaxis()
        ax.set_xlabel("Psycholinguistic Difficulty Score (lower = clearer)", fontsize=9.5)
        ax.set_title("Difficulty & Max Contradiction", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        for bar, v, c in zip(bars, vals, contras):
            ax.text(
                v + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} (contra={c:.2f})",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _format_lora_label(base_model_name: str) -> str:
    lower = base_model_name.lower()
    if "3b" in lower:
        return "Qwen2.5-3B LoRA"
    if "1.5b" in lower:
        return "Qwen2.5-1.5B LoRA"
    if "7b" in lower:
        return "Qwen2.5-7B LoRA"
    return "LoRA Single-Pass"


def _append_main_table(
    md_lines: list[str],
    title: str,
    summaries: list[dict[str, Any]],
    human_ref: dict[str, float],
    ref_label: str,
) -> None:
    md_lines.extend(
        [
            title,
            "",
            "| Method | Diff Score ↓ | Unatt. Bigram ↓ | Trigram PPL ↓ | Mean Dep Len ↓ | Red. ORC /100w ↓ | Stacked N /100w ↓ | Denom. V /100w ↓ | Inanim. Subj /100w ↓ | Cos Sim ↑ | NLI Min ↑ | NLI Contra ↓ | GPT-2 PPL ↓ | Code Pres ↑ | Latency (ms) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in summaries:
        md_lines.append(
            f"| {r['method']} | {r['difficulty_score']:.1f} | {r['unattested_bigram_rate']:.3f} | "
            f"{r['trigram_perplexity']:.1f} | {r['mean_dep_length']:.2f} | {r['reduced_orc_per_100w']:.2f} | "
            f"{r['stacked_noun_per_100w']:.2f} | {r['denominal_verb_per_100w']:.2f} | "
            f"{r['inanimate_subj_per_100w']:.2f} | {r['cosine_similarity']:.3f} | "
            f"{r['nli_min_entailment']:.3f} | {r['nli_max_contradiction']:.3f} | "
            f"{r['gpt2_perplexity']:.1f} | {r['code_preservation']:.3f} | {r['latency_ms']:.0f} |"
        )
    md_lines.append(
        f"| *{ref_label}* | *{human_ref['difficulty_score']:.1f}* | *{human_ref['unattested_bigram_rate']:.3f}* | "
        f"*{human_ref['trigram_perplexity']:.1f}* | *{human_ref['mean_dep_length']:.2f}* | *{human_ref['reduced_orc_per_100w']:.2f}* | "
        f"*{human_ref['stacked_noun_per_100w']:.2f}* | *{human_ref['denominal_verb_per_100w']:.2f}* | "
        f"*{human_ref['inanimate_subj_per_100w']:.2f}* | *1.000* | *1.000* | *0.000* | *{human_ref['gpt2_perplexity']:.1f}* | *1.000* | *-* |"
    )


def _append_domain_table(
    md_lines: list[str],
    title: str,
    domain_breakdowns: list[dict[str, Any]],
) -> None:
    md_lines.extend(
        [
            "",
            title,
            "",
            "| Method | Chat Diff ↓ | Chat NLI ↑ | Coding Diff ↓ | Coding NLI ↑ | Coding Code ↑ | Essay Diff ↓ | Essay NLI ↑ |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in domain_breakdowns:
        c = row.get("chat", {})
        k = row.get("coding", {})
        e = row.get("essay", {})
        md_lines.append(
            f"| {row['method']} | {c.get('difficulty_score', 0.0):.1f} | {c.get('nli_min_entailment', 0.0):.3f} | "
            f"{k.get('difficulty_score', 0.0):.1f} | {k.get('nli_min_entailment', 0.0):.3f} | {k.get('code_preservation', 0.0):.3f} | "
            f"{e.get('difficulty_score', 0.0):.1f} | {e.get('nli_min_entailment', 0.0):.3f} |"
        )


def _append_human_control_table(
    md_lines: list[str],
    title: str,
    summaries: list[dict[str, Any]],
) -> None:
    md_lines.extend(
        [
            "",
            title,
            "",
            "| Method | Diff Score | Cos Sim ↑ | NLI Min ↑ | NLI Contra ↓ | Code Pres ↑ | Length Ratio |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in summaries:
        md_lines.append(
            f"| {r['method']} | {r['difficulty_score']:.1f} | {r['cosine_similarity']:.3f} | "
            f"{r['nli_min_entailment']:.3f} | {r['nli_max_contradiction']:.3f} | "
            f"{r['code_preservation']:.3f} | {r['length_ratio']:.2f} |"
        )


def run_benchmark(
    results_dir: Path = Path("results"),
    adapter_dir: Path | None = Path("checkpoints/qwen25-3b-psycho-rev-lora"),
    include_neural: bool = True,
    skip_nllb: bool = False,
    skip_multipara: bool = False,
    nli_model: str | None = None,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    analyzer = PsycholinguisticAnalyzer()
    semantics = SemanticEvaluator(nli_model=nli_model) if nli_model else SemanticEvaluator()
    classical = ClassicalReverser()

    all_rows: list[dict[str, Any]] = []
    summaries_dense: list[dict[str, Any]] = []
    domain_dense: list[dict[str, Any]] = []
    summaries_human: list[dict[str, Any]] = []

    summaries_multipara: list[dict[str, Any]] = []
    domain_multipara: list[dict[str, Any]] = []
    summaries_human_multipara: list[dict[str, Any]] = []

    def _run_and_record(
        name: str,
        fn: Callable[[str], str],
        extra_latency_fn: Callable[[str], float] | None = None,
    ) -> None:
        print(f"=== Running method: {name} ===")
        dense_rows = _evaluate_method(
            name, TEST_BENCHMARK, fn, analyzer, semantics, split="single", extra_latency_fn=extra_latency_fn
        )
        human_rows = _evaluate_method(
            name, HUMAN_CONTROL, fn, analyzer, semantics, split="single", extra_latency_fn=extra_latency_fn
        )
        all_rows.extend(dense_rows)
        all_rows.extend(human_rows)

        s_dense = {"method": name, **_summarize_rows(dense_rows)}
        s_human = {"method": name, **_summarize_rows(human_rows)}
        summaries_dense.append(s_dense)
        domain_dense.append({"method": name, **_summarize_by_domain(dense_rows)})
        summaries_human.append(s_human)
        print(
            f"  [Single Dense] diff={s_dense['difficulty_score']:.1f} "
            f"cos={s_dense['cosine_similarity']:.3f} nli={s_dense['nli_min_entailment']:.3f} "
            f"contra={s_dense['nli_max_contradiction']:.3f} ppl={s_dense['gpt2_perplexity']:.1f} "
            f"code={s_dense['code_preservation']:.3f} lat={s_dense['latency_ms']:.0f}ms"
        )

        if not skip_multipara:
            mp_dense_rows = _evaluate_method(
                name,
                MULTIPARA_BENCHMARK,
                fn,
                analyzer,
                semantics,
                split="multipara",
                extra_latency_fn=extra_latency_fn,
            )
            mp_human_rows = _evaluate_method(
                name,
                HUMAN_CONTROL_MULTIPARA,
                fn,
                analyzer,
                semantics,
                split="multipara",
                extra_latency_fn=extra_latency_fn,
            )
            all_rows.extend(mp_dense_rows)
            all_rows.extend(mp_human_rows)

            s_mp = {"method": name, **_summarize_rows(mp_dense_rows)}
            s_mp_h = {"method": name, **_summarize_rows(mp_human_rows)}
            summaries_multipara.append(s_mp)
            domain_multipara.append({"method": name, **_summarize_by_domain(mp_dense_rows)})
            summaries_human_multipara.append(s_mp_h)
            print(
                f"  [Multi-Para]   diff={s_mp['difficulty_score']:.1f} "
                f"cos={s_mp['cosine_similarity']:.3f} nli={s_mp['nli_min_entailment']:.3f} "
                f"contra={s_mp['nli_max_contradiction']:.3f} ppl={s_mp['gpt2_perplexity']:.1f} "
                f"code={s_mp['code_preservation']:.3f} lat={s_mp['latency_ms']:.0f}ms"
            )

    # 1. Original baseline
    _run_and_record("Original", lambda x: x)

    # 2. Classical rule-based reverser
    _run_and_record("Classical (spaCy)", classical.rewrite)

    if include_neural:
        # 3. NLLB-200 Round-Trip Translation (En -> Ja -> En)
        if not skip_nllb:
            nllb = NLLBRoundTripReverser("facebook/nllb-200-distilled-600M")
            _run_and_record("NLLB-200 RTT", nllb.rewrite)
            nllb.close()

        # 4. Qwen2.5-3B-Instruct Zero-Shot + Best-of-6 Psycholinguistic Reranking
        qwen = QwenZeroShotReverser("Qwen/Qwen2.5-3B-Instruct")
        _run_and_record("Qwen2.5-3B ZeroShot", qwen.rewrite)

        reranker = BestOfNPsycholinguisticReverser(
            qwen, analyzer, semantics, num_samples=6, classical_fallback=classical
        )
        bon_cache: dict[str, tuple[str, float]] = {}

        def _bon_cached(text: str) -> str:
            if text not in bon_cache:
                t0 = time.perf_counter()
                rew = reranker.rewrite(text)
                bon_cache[text] = (rew, (time.perf_counter() - t0) * 1000.0)
            return bon_cache[text][0]

        _run_and_record("Qwen2.5-3B Best-of-6", _bon_cached)
        _run_and_record(
            "Hybrid (Best-of-6+Classical)",
            lambda x: classical.rewrite(bon_cache[x][0]),
            extra_latency_fn=lambda x: bon_cache[x][1],
        )
        qwen.close()

        # 5. Fine-tuned LoRA Single-Pass + Hybrid (auto-detects base model from adapter_config.json)
        if adapter_dir is not None and Path(adapter_dir).exists():
            lora = LoRASinglePassReverser(adapter_dir, base_model=None, classical_fallback=classical)
            lora_label = _format_lora_label(lora.base_model_name)
            lora_cache: dict[str, tuple[str, float]] = {}

            def _lora_cached(text: str) -> str:
                if text not in lora_cache:
                    t0 = time.perf_counter()
                    rew = lora.rewrite(text)
                    lora_cache[text] = (rew, (time.perf_counter() - t0) * 1000.0)
                return lora_cache[text][0]

            _run_and_record(lora_label, _lora_cached)
            _run_and_record(
                "Hybrid (LoRA+Classical)",
                lambda x: classical.rewrite(lora_cache[x][0]),
                extra_latency_fn=lambda x: lora_cache[x][1],
            )
            lora.close()

    human_ref = _summarize_rows(
        [r for r in all_rows if r["method"] == "Original" and r["split"] == "single" and r["domain"] == "human_control"]
    )
    human_ref_mp = (
        _summarize_rows(
            [
                r
                for r in all_rows
                if r["method"] == "Original" and r["split"] == "multipara" and r["domain"] == "human_control"
            ]
        )
        if not skip_multipara
        else {}
    )

    _plot_comparison(
        summaries_dense=summaries_dense,
        domain_breakdowns=domain_dense,
        summaries_multipara=summaries_multipara,
        human_ref=human_ref,
        out_path=results_dir / "comparison_plot.png",
    )

    payload: dict[str, Any] = {
        "dense_ai_summary": summaries_dense,
        "dense_ai_by_domain": domain_dense,
        "human_control_summary": summaries_human,
        "human_reference_baseline": human_ref,
        "multipara_summary": summaries_multipara,
        "multipara_by_domain": domain_multipara,
        "human_control_multipara_summary": summaries_human_multipara,
        "human_reference_multipara_baseline": human_ref_mp,
        "per_sample": all_rows,
    }
    with open(results_dir / "benchmark_results.json", "w") as f:
        json.dump(payload, f, indent=2)

    md_lines: list[str] = ["# Psycholinguistic Reversal Benchmark Results", ""]
    _append_main_table(
        md_lines,
        f"## Dense AI Single-Sentence Test Set ({len(TEST_BENCHMARK)} passages: Chat, Coding, Essay)",
        summaries_dense,
        human_ref,
        f"Human Reference ({len(HUMAN_CONTROL)} control)",
    )
    _append_domain_table(md_lines, "### Single-Sentence Per-Domain Breakdown", domain_dense)
    _append_human_control_table(
        md_lines,
        f"## Human Control Set — Single-Sentence ({len(HUMAN_CONTROL)} natural passages)",
        summaries_human,
    )

    if not skip_multipara and summaries_multipara:
        md_lines.append("")
        _append_main_table(
            md_lines,
            f"## Dense AI Multi-Paragraph Test Set ({len(MULTIPARA_BENCHMARK)} passages: Chat, Coding, Essay)",
            summaries_multipara,
            human_ref_mp,
            f"Human Reference Multi-Para ({len(HUMAN_CONTROL_MULTIPARA)} control)",
        )
        _append_domain_table(md_lines, "### Multi-Paragraph Per-Domain Breakdown", domain_multipara)
        _append_human_control_table(
            md_lines,
            f"## Human Control Set — Multi-Paragraph ({len(HUMAN_CONTROL_MULTIPARA)} natural passages)",
            summaries_human_multipara,
        )

    with open(results_dir / "summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m psycho_rev.evaluate",
        description="Run single-sentence and multi-paragraph psycholinguistic reversal benchmarks.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Output directory for results.")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("checkpoints/qwen25-3b-psycho-rev-lora"),
        help="Path to fine-tuned LoRA adapter directory (auto-detects base model).",
    )
    parser.add_argument("--classical-only", action="store_true", help="Skip all neural models.")
    parser.add_argument("--skip-nllb", action="store_true", help="Skip NLLB-200 round-trip translation baseline.")
    parser.add_argument("--skip-multipara", action="store_true", help="Skip multi-paragraph benchmark split.")
    parser.add_argument("--nli-model", default=None, help="Optional NLI cross-encoder model override.")
    args = parser.parse_args()

    run_benchmark(
        results_dir=args.results_dir,
        adapter_dir=args.adapter_dir,
        include_neural=not args.classical_only,
        skip_nllb=args.skip_nllb,
        skip_multipara=args.skip_multipara,
        nli_model=args.nli_model,
    )


if __name__ == "__main__":
    main()
