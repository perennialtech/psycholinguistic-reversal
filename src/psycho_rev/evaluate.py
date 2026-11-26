"""End-to-end benchmark runner comparing classical and neural psycholinguistic reversal methods."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

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
from psycho_rev.judge import BlindPairwiseJudge
from psycho_rev.neural import (
    BestOfNPsycholinguisticReverser,
    LoRASinglePassReverser,
    NLLBRoundTripReverser,
    QwenZeroShotReverser,
)
from psycho_rev.semantics import (
    DEFAULT_EVAL_NLI_MODEL,
    DEFAULT_HELDOUT_FLUENCY_MODEL,
    HeldOutSemanticEvaluator,
    SemanticEvaluator,
)

DOMAINS: tuple[str, ...] = ("chat", "coding", "essay")

def _select_evaluator_device() -> str:
    """Use CPU for auxiliary evaluators on <=12 GB GPUs so 3B generators never OOM."""
    try:
        import torch
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_gb >= 14.0:
                return "cuda"
    except Exception:
        pass
    return "cpu"



@dataclass(frozen=True)
class EvalSample:
    """Unified evaluation sample supporting legacy splits and external JSONL datasets."""

    sample_id: str
    domain: str
    text: str
    is_out_of_lexicon: bool | None = None


def load_jsonl_dataset(path: Path) -> tuple[list[EvalSample], list[EvalSample]]:
    """Load an external JSONL dataset into ``(test_samples, human_control_samples)``.

    Supports ``text`` (or ``original``/``passage``/``input``), ``sample_id`` (or ``id``),
    ``domain`` (or ``source``/``category``), ``is_out_of_lexicon`` (or ``out_of_lexicon``),
    and optional ``is_human_control`` / ``domain == "human_control"`` separation.
    """
    dataset_path = Path(path)
    test_samples: list[EvalSample] = []
    control_samples: list[EvalSample] = []

    with open(dataset_path, encoding="utf-8") as fh:
        for idx, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"Line {idx} in {dataset_path} is not a JSON object.")

            text = (
                rec.get("text")
                or rec.get("original")
                or rec.get("passage")
                or rec.get("input")
                or ""
            )
            if not isinstance(text, str) or not text.strip():
                continue

            sample_id = str(rec.get("sample_id") or rec.get("id") or f"ext_{idx:04d}")
            domain = str(rec.get("domain") or rec.get("source") or rec.get("category") or "external")

            raw_ool = rec.get("is_out_of_lexicon")
            if raw_ool is None and "out_of_lexicon" in rec:
                raw_ool = rec["out_of_lexicon"]
            is_ool = bool(raw_ool) if raw_ool is not None else None

            is_control = (
                domain.lower() == "human_control"
                or bool(rec.get("is_human_control", False))
                or str(rec.get("split", "")).lower() == "human_control"
            )
            sample = EvalSample(
                sample_id=sample_id,
                domain="human_control" if is_control else domain,
                text=text.strip(),
                is_out_of_lexicon=is_ool,
            )
            if is_control:
                control_samples.append(sample)
            else:
                test_samples.append(sample)

    return test_samples, control_samples


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
        "acceptability_score",
        "acceptability_mean",
        "acceptability_min",
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
    seen_domains: list[str] = list(DOMAINS)
    for r in rows:
        dom = r.get("domain")
        if isinstance(dom, str) and dom and dom != "human_control" and dom not in seen_domains:
            seen_domains.append(dom)
    for dom in seen_domains:
        dom_rows = [r for r in rows if r.get("domain") == dom]
        if dom_rows:
            out[dom] = _summarize_rows(dom_rows)
    return out


def _discover_domains(domain_breakdowns: list[dict[str, Any]]) -> list[str]:
    """Return all non-empty domain keys present across breakdown rows in stable order."""
    discovered: list[str] = []
    for row in domain_breakdowns:
        for key, val in row.items():
            if key == "method" or not isinstance(val, dict) or not val:
                continue
            if key not in discovered:
                discovered.append(key)
    return discovered


def _summarize_out_of_lexicon(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Break down summary metrics into out_of_lexicon vs in_lexicon buckets when flags exist."""
    ool_rows = [r for r in rows if r.get("is_out_of_lexicon") is True]
    in_rows = [r for r in rows if r.get("is_out_of_lexicon") is False]
    if not ool_rows and not in_rows:
        return {}
    out: dict[str, dict[str, Any]] = {}
    if ool_rows:
        out["out_of_lexicon"] = {"count": len(ool_rows), **_summarize_rows(ool_rows)}
    if in_rows:
        out["in_lexicon"] = {"count": len(in_rows), **_summarize_rows(in_rows)}
    return out


def _generate_outputs(
    samples: Sequence[BenchmarkSample | EvalSample],
    rewriter_fn: Callable[[str], str],
    extra_latency_fn: Callable[[str], float] | None = None,
) -> tuple[list[str], list[float]]:
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
    return rewrittens, latencies


def _score_outputs(
    name: str,
    samples: Sequence[BenchmarkSample | EvalSample],
    rewrittens: Sequence[str],
    latencies: Sequence[float],
    analyzer: PsycholinguisticAnalyzer,
    semantics: SemanticEvaluator,
    split: str = "single",
) -> list[dict[str, Any]]:
    originals = [s.text for s in samples]
    sem_reports = semantics.evaluate_batch(originals, list(rewrittens))
    rows: list[dict[str, Any]] = []
    for sample, rew, lat, sem in zip(samples, rewrittens, latencies, sem_reports):
        feat = analyzer.analyze(rew)
        sem_dict = sem.to_dict()
        sem_dict.setdefault("nli_max_contradiction", 0.0)
        sem_dict.setdefault("acceptability_mean", 1.0)
        sem_dict.setdefault("acceptability_min", 1.0)
        sem_dict.setdefault("acceptability_score", sem_dict["acceptability_min"])
        row: dict[str, Any] = {
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
        is_ool = getattr(sample, "is_out_of_lexicon", None)
        if is_ool is not None:
            row["is_out_of_lexicon"] = bool(is_ool)
        rows.append(row)
    return rows


def _evaluate_method(
    name: str,
    samples: Sequence[BenchmarkSample | EvalSample],
    rewriter_fn: Callable[[str], str],
    analyzer: PsycholinguisticAnalyzer,
    semantics: SemanticEvaluator,
    split: str = "single",
    extra_latency_fn: Callable[[str], float] | None = None,
) -> list[dict[str, Any]]:
    rewrittens, latencies = _generate_outputs(samples, rewriter_fn, extra_latency_fn=extra_latency_fn)
    return _score_outputs(name, samples, rewrittens, latencies, analyzer, semantics, split=split)

def _plot_comparison(
    summaries_dense: list[dict[str, Any]],
    domain_breakdowns: list[dict[str, Any]],
    summaries_multipara: list[dict[str, Any]],
    human_ref: dict[str, float] | None,
    out_path: Path,
) -> None:
    if not summaries_dense:
        return

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
    if human_ref and "difficulty_score" in human_ref:
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
        ax.annotate(
            "Human Reference",
            (human_ref["difficulty_score"], 1.0),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8,
            fontweight="bold",
            color="#854d0e",
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
    active_domains = _discover_domains(domain_breakdowns)

    if active_domains:
        w = 0.75 / max(len(active_domains), 1)
        dom_palette = [
            "#2563eb",
            "#059669",
            "#d97706",
            "#7c3aed",
            "#dc2626",
            "#0f766e",
            "#be123c",
            "#64748b",
        ]
        offsets = np.linspace(
            -w * (len(active_domains) - 1) / 2.0,
            w * (len(active_domains) - 1) / 2.0,
            len(active_domains),
        )
        for idx_d, (offset, dom) in enumerate(zip(offsets, active_domains)):
            vals = [dom_map.get(m, {}).get(dom, {}).get("nli_min_entailment", 0.0) for m in methods]
            ax.bar(
                x + offset,
                vals,
                width=w,
                label=dom.replace("_", " ").title(),
                color=dom_palette[idx_d % len(dom_palette)],
                alpha=0.9,
            )
        ax.legend(fontsize=8.5, frameon=True, loc="lower right")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=28, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Bidirectional NLI Min-Entailment", fontsize=9.5)
    ax.set_title("Domain NLI Fidelity Breakdown", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel 3: Single-Sentence vs Multi-Paragraph (or Acceptability Min when single-split)
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
        acc_mins = dense_df["acceptability_min"].tolist() if "acceptability_min" in dense_df else [1.0] * len(methods)
        bars = ax.barh(methods, vals, color=colors, edgecolor="none", height=0.6)
        ax.invert_yaxis()
        ax.set_xlabel("Psycholinguistic Difficulty Score (lower = clearer)", fontsize=9.5)
        ax.set_title("Difficulty & CoLA Acceptability Min", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        for bar, v, acc in zip(bars, vals, acc_mins):
            ax.text(
                v + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f} (acc_min={acc:.2f})",
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
    human_ref: dict[str, float] | None,
    ref_label: str,
) -> None:
    md_lines.extend(
        [
            title,
            "",
            "| Method | Diff Score ↓ | Unatt. Bigram ↓ | Trigram PPL ↓ | Mean Dep Len ↓ | Red. ORC /100w ↓ | Stacked N /100w ↓ | Denom. V /100w ↓ | Inanim. Subj /100w ↓ | Cos Sim ↑ | NLI Min ↑ | NLI Contra ↓ | GPT-2 PPL ↓ | Acc Mean ↑ | Acc Min ↑ | Code Pres ↑ | Latency (ms) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in summaries:
        md_lines.append(
            f"| {r['method']} | {r['difficulty_score']:.1f} | {r['unattested_bigram_rate']:.3f} | "
            f"{r['trigram_perplexity']:.1f} | {r['mean_dep_length']:.2f} | {r['reduced_orc_per_100w']:.2f} | "
            f"{r['stacked_noun_per_100w']:.2f} | {r['denominal_verb_per_100w']:.2f} | "
            f"{r['inanimate_subj_per_100w']:.2f} | {r['cosine_similarity']:.3f} | "
            f"{r['nli_min_entailment']:.3f} | {r['nli_max_contradiction']:.3f} | "
            f"{r['gpt2_perplexity']:.1f} | {r.get('acceptability_mean', 1.0):.3f} | "
            f"{r.get('acceptability_min', 1.0):.3f} | {r['code_preservation']:.3f} | {r['latency_ms']:.0f} |"
        )
    if human_ref and "difficulty_score" in human_ref:
        md_lines.append(
            f"| *{ref_label}* | *{human_ref['difficulty_score']:.1f}* | *{human_ref['unattested_bigram_rate']:.3f}* | "
            f"*{human_ref['trigram_perplexity']:.1f}* | *{human_ref['mean_dep_length']:.2f}* | *{human_ref['reduced_orc_per_100w']:.2f}* | "
            f"*{human_ref['stacked_noun_per_100w']:.2f}* | *{human_ref['denominal_verb_per_100w']:.2f}* | "
            f"*{human_ref['inanimate_subj_per_100w']:.2f}* | *1.000* | *1.000* | *0.000* | "
            f"*{human_ref['gpt2_perplexity']:.1f}* | *{human_ref.get('acceptability_mean', 1.0):.3f}* | "
            f"*{human_ref.get('acceptability_min', 1.0):.3f}* | *1.000* | *-* |"
        )


def _append_domain_table(
    md_lines: list[str],
    title: str,
    domain_breakdowns: list[dict[str, Any]],
) -> None:
    if not domain_breakdowns:
        return
    active_domains = _discover_domains(domain_breakdowns)
    if not active_domains:
        return
    if set(active_domains) == set(DOMAINS):
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
        return

    header = "| Method | " + " | ".join(f"{d} Diff ↓ | {d} NLI ↑ | {d} Acc Min ↑" for d in active_domains) + " |"
    sep = "| :--- | " + " | ".join("---: | ---: | ---:" for _ in active_domains) + " |"
    md_lines.extend(["", title, "", header, sep])
    for row in domain_breakdowns:
        cells = []
        for d in active_domains:
            sub = row.get(d, {})
            cells.append(
                f"{sub.get('difficulty_score', 0.0):.1f} | {sub.get('nli_min_entailment', 0.0):.3f} | {sub.get('acceptability_min', 1.0):.3f}"
            )
        md_lines.append(f"| {row['method']} | " + " | ".join(cells) + " |")


def _append_out_of_lexicon_table(
    md_lines: list[str],
    title: str,
    ool_breakdowns: list[dict[str, Any]],
) -> None:
    if not ool_breakdowns:
        return
    md_lines.extend(
        [
            "",
            title,
            "",
            "| Method | OOL Diff ↓ | OOL NLI ↑ | OOL Acc Mean ↑ | OOL Acc Min ↑ | In-Lex Diff ↓ | In-Lex NLI ↑ | In-Lex Acc Mean ↑ | In-Lex Acc Min ↑ |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in ool_breakdowns:
        ool = row.get("out_of_lexicon", {})
        inl = row.get("in_lexicon", {})
        md_lines.append(
            f"| {row['method']} | {ool.get('difficulty_score', 0.0):.1f} | {ool.get('nli_min_entailment', 0.0):.3f} | "
            f"{ool.get('acceptability_mean', 0.0):.3f} | {ool.get('acceptability_min', 0.0):.3f} | "
            f"{inl.get('difficulty_score', 0.0):.1f} | {inl.get('nli_min_entailment', 0.0):.3f} | "
            f"{inl.get('acceptability_mean', 0.0):.3f} | {inl.get('acceptability_min', 0.0):.3f} |"
        )


def _append_human_control_table(
    md_lines: list[str],
    title: str,
    summaries: list[dict[str, Any]],
) -> None:
    if not summaries:
        return
    md_lines.extend(
        [
            "",
            title,
            "",
            "| Method | Diff Score | Cos Sim ↑ | NLI Min ↑ | NLI Contra ↓ | Acc Mean ↑ | Acc Min ↑ | Code Pres ↑ | Length Ratio |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in summaries:
        md_lines.append(
            f"| {r['method']} | {r['difficulty_score']:.1f} | {r['cosine_similarity']:.3f} | "
            f"{r['nli_min_entailment']:.3f} | {r['nli_max_contradiction']:.3f} | "
            f"{r.get('acceptability_mean', 1.0):.3f} | {r.get('acceptability_min', 1.0):.3f} | "
            f"{r['code_preservation']:.3f} | {r['length_ratio']:.2f} |"
        )


def _append_blind_judge_table(
    md_lines: list[str],
    title: str,
    judge_summaries: list[dict[str, Any]],
    baseline_name: str,
) -> None:
    if not judge_summaries:
        return
    md_lines.extend(
        [
            "",
            title,
            "",
            f"| Method (vs {baseline_name}) | Pairs | Win % ↑ | Tie % | Loss % ↓ | Agree % | Clarity ↑ | Naturalness ↑ | Meaning ↑ |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for s in judge_summaries:
        ma = s.get("mean_scores_a", {})
        md_lines.append(
            f"| {s['method_a']} | {s['total_pairs']} | {s['win_rate_a'] * 100:.1f}% | "
            f"{s['tie_rate'] * 100:.1f}% | {s['win_rate_b'] * 100:.1f}% | "
            f"{s['agreement_rate'] * 100:.1f}% | {ma.get('clarity', 0.0):.2f} | "
            f"{ma.get('naturalness', 0.0):.2f} | {ma.get('meaning_preservation', 0.0):.2f} |"
        )


def _run_blind_judge_comparisons(
    all_rows: list[dict[str, Any]],
    *,
    primary_split: str,
    baseline_method: str,
    judge_model: str | None,
    judge_max_samples: int,
) -> list[dict[str, Any]]:
    """Run blind position-swapped pairwise LLM judge comparing each method vs baseline_method."""
    split_rows = [
        r for r in all_rows if r.get("split") == primary_split and r.get("domain") != "human_control"
    ]
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    method_order: list[str] = []
    for row in split_rows:
        m = row["method"]
        if m not in by_method:
            by_method[m] = {}
            method_order.append(m)
        by_method[m][row["sample_id"]] = row

    if baseline_method not in by_method:
        return []

    base_map = by_method[baseline_method]
    sample_ids = list(base_map.keys())[: max(1, int(judge_max_samples))]
    summaries: list[dict[str, Any]] = []

    with BlindPairwiseJudge(model=judge_model) as judge:
        for method in method_order:
            if method == baseline_method:
                continue
            cand_map = by_method[method]
            pairs = [
                (
                    sid,
                    base_map[sid]["original"],
                    cand_map[sid]["rewritten"],
                    base_map[sid]["rewritten"],
                )
                for sid in sample_ids
                if sid in cand_map
            ]
            if not pairs:
                continue
            summary = judge.evaluate_pairs(pairs, method_a=method, method_b=baseline_method)
            summaries.append(summary.to_dict())
            print(
                f"  [Blind Judge: {method} vs {baseline_method}] "
                f"win={summary.win_rate_a * 100:.1f}% tie={summary.tie_rate * 100:.1f}% "
                f"loss={summary.win_rate_b * 100:.1f}% agree={summary.agreement_rate * 100:.1f}%"
            )

    return summaries


def run_benchmark(
    results_dir: Path = Path("results"),
    adapter_dir: Path | None = Path("checkpoints/qwen25-3b-psycho-rev-lora"),
    include_neural: bool = True,
    skip_nllb: bool = False,
    skip_multipara: bool = False,
    nli_model: str | None = None,
    *,
    dataset_path: Path | None = None,
    lm_model: str | None = None,
    enable_acceptability: bool = True,
    blind_judge: bool = False,
    judge_model: str | None = None,
    judge_baseline: str = "Original",
    judge_max_samples: int = 20,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    analyzer = PsycholinguisticAnalyzer()
    eval_device = _select_evaluator_device()
    classical = ClassicalReverser()

    use_external = dataset_path is not None
    if use_external:
        ext_test_samples, ext_control_samples = load_jsonl_dataset(Path(dataset_path))
        primary_samples: Sequence[BenchmarkSample | EvalSample] = ext_test_samples
        control_samples: Sequence[BenchmarkSample | EvalSample] = ext_control_samples
        primary_split = "external"
        effective_skip_multipara = True
    else:
        primary_samples = TEST_BENCHMARK
        control_samples = HUMAN_CONTROL
        primary_split = "single"
        effective_skip_multipara = skip_multipara

    all_rows: list[dict[str, Any]] = []
    summaries_dense: list[dict[str, Any]] = []
    domain_dense: list[dict[str, Any]] = []
    ool_dense: list[dict[str, Any]] = []
    summaries_human: list[dict[str, Any]] = []

    summaries_multipara: list[dict[str, Any]] = []
    domain_multipara: list[dict[str, Any]] = []
    summaries_human_multipara: list[dict[str, Any]] = []

    # Phase 1: generate all outputs, closing each neural generator before loading the next.
    generated_runs: list[dict[str, Any]] = []

    def _run_and_record(
        name: str,
        fn: Callable[[str], str],
        extra_latency_fn: Callable[[str], float] | None = None,
    ) -> None:
        print(f"=== Generating outputs: {name} ===")
        p_rew, p_lat = _generate_outputs(primary_samples, fn, extra_latency_fn=extra_latency_fn)
        c_rew, c_lat = (
            _generate_outputs(control_samples, fn, extra_latency_fn=extra_latency_fn)
            if control_samples
            else ([], [])
        )
        mp_rew, mp_lat = (
            _generate_outputs(MULTIPARA_BENCHMARK, fn, extra_latency_fn=extra_latency_fn)
            if not effective_skip_multipara
            else ([], [])
        )
        mph_rew, mph_lat = (
            _generate_outputs(HUMAN_CONTROL_MULTIPARA, fn, extra_latency_fn=extra_latency_fn)
            if not effective_skip_multipara
            else ([], [])
        )
        generated_runs.append(
            {
                "name": name,
                "primary": (p_rew, p_lat),
                "control": (c_rew, c_lat),
                "mp": (mp_rew, mp_lat),
                "mp_human": (mph_rew, mph_lat),
            }
        )

    # 1. Original baseline
    _run_and_record("Original", lambda x: x)

    # 2. Classical rule-based reverser
    _run_and_record("Classical (spaCy)", classical.rewrite)

    if include_neural:
        # 3. NLLB-200 Round-Trip Translation (En -> Ja -> En)
        if not skip_nllb:
            nllb = NLLBRoundTripReverser("facebook/nllb-200-distilled-600M")
            try:
                _run_and_record("NLLB-200 RTT", nllb.rewrite)
            finally:
                nllb.close()

        # 4. Qwen2.5-3B-Instruct Zero-Shot + Best-of-6 Psycholinguistic Reranking
        qwen = QwenZeroShotReverser("Qwen/Qwen2.5-3B-Instruct")
        try:
            _run_and_record("Qwen2.5-3B ZeroShot", qwen.rewrite)

            rerank_semantics = SemanticEvaluator(device=eval_device, role="rerank", enable_acceptability=False)
            try:
                reranker = BestOfNPsycholinguisticReverser(
                    qwen, analyzer, rerank_semantics, num_samples=6, classical_fallback=classical
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
            finally:
                rerank_semantics.close()
        finally:
            qwen.close()

        # 5. Fine-tuned LoRA Single-Pass + Hybrid
        if adapter_dir is not None and Path(adapter_dir).exists():
            lora = LoRASinglePassReverser(adapter_dir, base_model=None, classical_fallback=classical)
            try:
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
            finally:
                lora.close()

    # Phase 2: score all generated outputs on GPU with zero generator VRAM overlap.
    semantics = HeldOutSemanticEvaluator(
        nli_model=nli_model or DEFAULT_EVAL_NLI_MODEL,
        lm_model=lm_model or DEFAULT_HELDOUT_FLUENCY_MODEL,
        enable_acceptability=enable_acceptability,
    )
    try:
        split_tag = "External" if use_external else "Single Dense"
        for run in generated_runs:
            name = run["name"]
            print(f"=== Scoring method: {name} ===")
            p_rew, p_lat = run["primary"]
            dense_rows = _score_outputs(name, primary_samples, p_rew, p_lat, analyzer, semantics, split=primary_split)
            all_rows.extend(dense_rows)

            s_dense = {"method": name, **_summarize_rows(dense_rows)}
            summaries_dense.append(s_dense)
            domain_dense.append({"method": name, **_summarize_by_domain(dense_rows)})
            ool_stats = _summarize_out_of_lexicon(dense_rows)
            if ool_stats:
                ool_dense.append({"method": name, **ool_stats})

            if control_samples:
                c_rew, c_lat = run["control"]
                human_rows = _score_outputs(name, control_samples, c_rew, c_lat, analyzer, semantics, split=primary_split)
                all_rows.extend(human_rows)
                s_human = {"method": name, **_summarize_rows(human_rows)}
                summaries_human.append(s_human)

            print(
                f"  [{split_tag}] diff={s_dense['difficulty_score']:.1f} "
                f"cos={s_dense['cosine_similarity']:.3f} nli={s_dense['nli_min_entailment']:.3f} "
                f"contra={s_dense['nli_max_contradiction']:.3f} ppl={s_dense['gpt2_perplexity']:.1f} "
                f"acc={s_dense['acceptability_mean']:.3f}/{s_dense['acceptability_min']:.3f} "
                f"code={s_dense['code_preservation']:.3f} lat={s_dense['latency_ms']:.0f}ms"
            )

            if not effective_skip_multipara:
                mp_rew, mp_lat = run["mp"]
                mph_rew, mph_lat = run["mp_human"]
                mp_dense_rows = _score_outputs(name, MULTIPARA_BENCHMARK, mp_rew, mp_lat, analyzer, semantics, split="multipara")
                mp_human_rows = _score_outputs(name, HUMAN_CONTROL_MULTIPARA, mph_rew, mph_lat, analyzer, semantics, split="multipara")
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
                    f"acc={s_mp['acceptability_mean']:.3f}/{s_mp['acceptability_min']:.3f} "
                    f"code={s_mp['code_preservation']:.3f} lat={s_mp['latency_ms']:.0f}ms"
                )
    finally:
        semantics.close()

    human_ref_rows = [
        r
        for r in all_rows
        if r["method"] == "Original" and r["split"] == primary_split and r["domain"] == "human_control"
    ]
    human_ref = _summarize_rows(human_ref_rows) if human_ref_rows else {}

    human_ref_mp = (
        _summarize_rows(
            [
                r
                for r in all_rows
                if r["method"] == "Original" and r["split"] == "multipara" and r["domain"] == "human_control"
            ]
        )
        if not effective_skip_multipara
        else {}
    )

    blind_judge_summary: list[dict[str, Any]] = []
    if blind_judge:
        blind_judge_summary = _run_blind_judge_comparisons(
            all_rows,
            primary_split=primary_split,
            baseline_method=judge_baseline,
            judge_model=judge_model,
            judge_max_samples=judge_max_samples,
        )

    _plot_comparison(
        summaries_dense=summaries_dense,
        domain_breakdowns=domain_dense,
        summaries_multipara=summaries_multipara,
        human_ref=human_ref,
        out_path=results_dir / "comparison_plot.png",
    )

    payload: dict[str, Any] = {
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "dense_ai_summary": summaries_dense,
        "dense_ai_by_domain": domain_dense,
        "out_of_lexicon_breakdown": ool_dense,
        "human_control_summary": summaries_human,
        "human_reference_baseline": human_ref,
        "multipara_summary": summaries_multipara,
        "multipara_by_domain": domain_multipara,
        "human_control_multipara_summary": summaries_human_multipara,
        "human_reference_multipara_baseline": human_ref_mp,
        "blind_judge_summary": blind_judge_summary,
        "per_sample": all_rows,
    }
    with open(results_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_lines: list[str] = ["# Psycholinguistic Reversal Benchmark Results", ""]
    primary_title = (
        f"## External Dataset Test Set ({len(primary_samples)} passages from `{Path(dataset_path).name}`)"
        if use_external and dataset_path is not None
        else f"## Dense AI Single-Sentence Test Set ({len(primary_samples)} passages: Chat, Coding, Essay)"
    )
    _append_main_table(
        md_lines,
        primary_title,
        summaries_dense,
        human_ref,
        f"Human Reference ({len(control_samples)} control)",
    )
    _append_domain_table(md_lines, "### Per-Domain Breakdown", domain_dense)
    if ool_dense:
        _append_out_of_lexicon_table(md_lines, "### Out-of-Lexicon vs In-Lexicon Breakdown", ool_dense)
    if summaries_human:
        _append_human_control_table(
            md_lines,
            f"## Human Control Set ({len(control_samples)} natural passages)",
            summaries_human,
        )

    if not effective_skip_multipara and summaries_multipara:
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

    if blind_judge_summary:
        _append_blind_judge_table(
            md_lines,
            f"## Blind Pairwise LLM-as-a-Judge (Position-Swapped vs {judge_baseline})",
            blind_judge_summary,
            judge_baseline,
        )

    with open(results_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m psycho_rev.evaluate",
        description="Run single-sentence, multi-paragraph, or external JSONL psycholinguistic reversal benchmarks.",
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
    parser.add_argument("--nli-model", default=None, help="Optional held-out NLI cross-encoder model override.")
    parser.add_argument(
        "--lm-model",
        default=None,
        help="Optional held-out LM perplexity model override (default: gpt2-medium).",
    )
    parser.add_argument(
        "--disable-acceptability",
        action="store_true",
        help="Disable CoLA grammatical acceptability scoring.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to external JSONL dataset to evaluate (supports is_out_of_lexicon breakdown).",
    )
    parser.add_argument(
        "--blind-judge",
        action="store_true",
        help="Run blind position-swapped pairwise LLM-as-a-judge against --judge-baseline.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model identifier for BlindPairwiseJudge (defaults to provider's gpt-4o-mini).",
    )
    parser.add_argument(
        "--judge-baseline",
        default="Original",
        help="Baseline method name to compare against when --blind-judge is enabled.",
    )
    parser.add_argument(
        "--judge-max-samples",
        type=int,
        default=20,
        help="Maximum number of samples per method pair for blind pairwise judging.",
    )
    args = parser.parse_args()

    run_benchmark(
        results_dir=args.results_dir,
        adapter_dir=args.adapter_dir,
        include_neural=not args.classical_only,
        skip_nllb=args.skip_nllb,
        skip_multipara=args.skip_multipara,
        nli_model=args.nli_model,
        dataset_path=args.dataset_path,
        lm_model=args.lm_model,
        enable_acceptability=not args.disable_acceptability,
        blind_judge=args.blind_judge,
        judge_model=args.judge_model,
        judge_baseline=args.judge_baseline,
        judge_max_samples=args.judge_max_samples,
    )


if __name__ == "__main__":
    main()
