"""Command-line interface for analyzing and reversing psycholinguistic difficulty features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from psycho_rev.classical import ClassicalReverser
from psycho_rev.features import PsycholinguisticAnalyzer


def _read_input_text(
    parser: argparse.ArgumentParser,
    text_arg: str | None,
    file_arg: str | None,
) -> str:
    if file_arg is not None:
        if file_arg == "-":
            return sys.stdin.read()
        return Path(file_arg).read_text(encoding="utf-8")
    if text_arg is not None:
        return text_arg
    if not sys.stdin.isatty():
        return sys.stdin.read()
    parser.error("Provide input text as a positional argument, via --file <path>, or via stdin (-).")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="psycho-rev",
        description="Analyze and reverse psycholinguistic difficulty features in LLM text.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="Compute all six psycholinguistic features for input text")
    p_an.add_argument("text", nargs="?", default=None, help="Text to analyze (or use --file / stdin)")
    p_an.add_argument("--file", dest="input_file", default=None, help="Path to input file, or '-' for stdin")

    p_rw = sub.add_parser("rewrite", help="Rewrite text using a classical, neural, or hybrid reverser")
    p_rw.add_argument("text", nargs="?", default=None, help="Text to rewrite (or use --file / stdin)")
    p_rw.add_argument("--file", dest="input_file", default=None, help="Path to input file, or '-' for stdin")
    p_rw.add_argument(
        "--plain",
        action="store_true",
        help="Emit only the rewritten text (suitable for Unix pipes) instead of JSON",
    )
    p_rw.add_argument(
        "--method",
        choices=["classical", "nllb", "qwen", "best-of-n", "lora", "hybrid-bon", "hybrid-lora"],
        default="classical",
    )
    p_rw.add_argument(
        "--adapter-dir",
        default="8q0sb/qwen25-3b-psycholinguistic-reversal-lora",
        help="LoRA adapter directory or Hugging Face repo ID",
    )
    p_rw.add_argument(
        "--base-model",
        default=None,
        help="Optional base model override (auto-detected from adapter_config.json for LoRA methods)",
    )
    p_rw.add_argument("--samples", type=int, default=6, help="Candidate count for best-of-n / hybrid-bon")

    p_bench = sub.add_parser("bench", help="Run the full multi-domain benchmark")
    p_bench.add_argument("--results-dir", default="results")
    p_bench.add_argument("--adapter-dir", default="checkpoints/qwen25-3b-psycho-rev-lora")
    p_bench.add_argument("--classical-only", action="store_true")
    p_bench.add_argument("--skip-nllb", action="store_true")
    p_bench.add_argument("--skip-multipara", action="store_true")
    p_bench.add_argument("--nli-model", default=None)

    args = parser.parse_args()

    if args.cmd == "analyze":
        text = _read_input_text(parser, args.text, args.input_file)
        analyzer = PsycholinguisticAnalyzer()
        print(json.dumps(analyzer.analyze(text).to_dict(), indent=2))
    elif args.cmd == "rewrite":
        text = _read_input_text(parser, args.text, args.input_file)
        analyzer = None if args.plain else PsycholinguisticAnalyzer()
        before = analyzer.analyze(text) if analyzer is not None else None

        if args.method == "classical":
            rew = ClassicalReverser().rewrite(text)
        elif args.method == "nllb":
            from psycho_rev.neural import NLLBRoundTripReverser

            model = NLLBRoundTripReverser()
            rew = model.rewrite(text)
            model.close()
        elif args.method == "qwen":
            from psycho_rev.neural import QwenZeroShotReverser

            model = QwenZeroShotReverser(model_name=args.base_model or "Qwen/Qwen2.5-3B-Instruct")
            rew = model.rewrite(text)
            model.close()
        elif args.method in ("best-of-n", "hybrid-bon"):
            from psycho_rev.neural import (
                BestOfNPsycholinguisticReverser,
                HybridReverser,
                QwenZeroShotReverser,
            )
            from psycho_rev.semantics import SemanticEvaluator

            an = analyzer or PsycholinguisticAnalyzer()
            gen = QwenZeroShotReverser(model_name=args.base_model or "Qwen/Qwen2.5-3B-Instruct")
            sem = SemanticEvaluator()
            bon = BestOfNPsycholinguisticReverser(gen, an, sem, num_samples=args.samples)
            if args.method == "hybrid-bon":
                rew = HybridReverser(bon).rewrite(text)
            else:
                rew = bon.rewrite(text)
            gen.close()
        else:
            from psycho_rev.neural import HybridReverser, LoRASinglePassReverser

            lora = LoRASinglePassReverser(args.adapter_dir, base_model=args.base_model)
            if args.method == "hybrid-lora":
                rew = HybridReverser(lora).rewrite(text)
            else:
                rew = lora.rewrite(text)
            lora.close()

        if args.plain:
            print(rew)
        else:
            assert analyzer is not None and before is not None
            after = analyzer.analyze(rew)
            print(
                json.dumps(
                    {
                        "original": text,
                        "rewritten": rew,
                        "before": before.difficulty_score,
                        "after": after.difficulty_score,
                    },
                    indent=2,
                )
            )
    elif args.cmd == "bench":
        from psycho_rev.evaluate import run_benchmark

        run_benchmark(
            results_dir=Path(args.results_dir),
            adapter_dir=Path(args.adapter_dir) if args.adapter_dir else None,
            include_neural=not args.classical_only,
            skip_nllb=args.skip_nllb,
            skip_multipara=args.skip_multipara,
            nli_model=args.nli_model,
        )


if __name__ == "__main__":
    main()
