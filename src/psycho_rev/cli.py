"""Command-line interface for analyzing and reversing psycholinguistic difficulty features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psycho_rev.classical import ClassicalReverser
from psycho_rev.features import PsycholinguisticAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="psycho-rev",
        description="Analyze and reverse psycholinguistic difficulty features in LLM text.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="Compute all six psycholinguistic features for input text")
    p_an.add_argument("text", help="Text to analyze")

    p_rw = sub.add_parser("rewrite", help="Rewrite text using a classical or neural reverser")
    p_rw.add_argument("text", help="Text to rewrite")
    p_rw.add_argument(
        "--method",
        choices=["classical", "nllb", "qwen", "best-of-n", "lora"],
        default="classical",
    )
    p_rw.add_argument("--adapter-dir", default="checkpoints/qwen25-1.5b-psycho-rev-lora")
    p_rw.add_argument("--samples", type=int, default=6)

    p_bench = sub.add_parser("bench", help="Run the full multi-domain benchmark")
    p_bench.add_argument("--results-dir", default="results")
    p_bench.add_argument("--adapter-dir", default="checkpoints/qwen25-1.5b-psycho-rev-lora")
    p_bench.add_argument("--classical-only", action="store_true")

    args = parser.parse_args()

    if args.cmd == "analyze":
        analyzer = PsycholinguisticAnalyzer()
        print(json.dumps(analyzer.analyze(args.text).to_dict(), indent=2))
    elif args.cmd == "rewrite":
        analyzer = PsycholinguisticAnalyzer()
        before = analyzer.analyze(args.text)
        if args.method == "classical":
            rew = ClassicalReverser().rewrite(args.text)
        elif args.method == "nllb":
            from psycho_rev.neural import NLLBRoundTripReverser
            model = NLLBRoundTripReverser()
            rew = model.rewrite(args.text)
            model.close()
        elif args.method == "qwen":
            from psycho_rev.neural import QwenZeroShotReverser
            model = QwenZeroShotReverser()
            rew = model.rewrite(args.text)
            model.close()
        elif args.method == "best-of-n":
            from psycho_rev.neural import BestOfNPsycholinguisticReverser, QwenZeroShotReverser
            from psycho_rev.semantics import SemanticEvaluator
            gen = QwenZeroShotReverser()
            sem = SemanticEvaluator()
            rew = BestOfNPsycholinguisticReverser(gen, analyzer, sem, num_samples=args.samples).rewrite(args.text)
            gen.close()
        else:
            from psycho_rev.neural import LoRASinglePassReverser
            model = LoRASinglePassReverser(args.adapter_dir)
            rew = model.rewrite(args.text)
            model.close()
        after = analyzer.analyze(rew)
        print(json.dumps({"original": args.text, "rewritten": rew, "before": before.difficulty_score, "after": after.difficulty_score}, indent=2))
    elif args.cmd == "bench":
        from psycho_rev.evaluate import run_benchmark
        run_benchmark(
            results_dir=Path(args.results_dir),
            adapter_dir=Path(args.adapter_dir) if args.adapter_dir else None,
            include_neural=not args.classical_only,
        )


if __name__ == "__main__":
    main()
