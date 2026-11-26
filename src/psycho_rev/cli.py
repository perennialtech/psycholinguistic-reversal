"""Command-line interface for analyzing and reversing psycholinguistic difficulty features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REWRITE_METHODS: tuple[str, ...] = (
    "openrouter",
    "openai",
    "lora",
    "best-of-n",
    "hybrid-bon",
    "hybrid-lora",
    "classical",
    "zeroshot",
    "rtt",
)


def _read_input_text(
    parser: argparse.ArgumentParser,
    text_arg: str | None,
    file_arg: str | None,
) -> tuple[str, str]:
    """Return ``(text, display_filename)`` from ``--file``, positional text, or stdin."""
    if file_arg is not None:
        if file_arg == "-":
            return sys.stdin.read(), "stdin"
        return Path(file_arg).read_text(encoding="utf-8"), file_arg
    if text_arg is not None:
        if text_arg == "-":
            return sys.stdin.read(), "stdin"
        return text_arg, "input"
    if not sys.stdin.isatty():
        return sys.stdin.read(), "stdin"
    parser.error("Provide input text as a positional argument, via --file <path>, or via stdin (-).")
    return "", "input"


def _format_analysis_text(report: dict[str, object]) -> str:
    lines = [
        f"Difficulty score:       {report.get('difficulty_score')}",
        f"Words / Sentences:      {report.get('word_count')} / {report.get('sentence_count')}",
        f"Unattested bigrams:     {report.get('unattested_bigram_rate')}",
        f"Trigram perplexity:     {report.get('trigram_perplexity')}",
        f"Mean / Std dep length:  {report.get('mean_dep_length')} / {report.get('std_dep_length')}",
        f"Reduced ORCs:           {report.get('reduced_orc_count')} ({report.get('reduced_orc_per_100w')}/100w)",
        f"Stacked noun compounds: {report.get('stacked_noun_count')} ({report.get('stacked_noun_per_100w')}/100w)",
        f"Denominal verbs:        {report.get('denominal_verb_count')} ({report.get('denominal_verb_per_100w')}/100w)",
        f"Inanimate subjects:     {report.get('inanimate_subj_count')} ({report.get('inanimate_subj_per_100w')}/100w)",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    from psycho_rev.rewriter import DEFAULT_LORA_ADAPTER

    parser = argparse.ArgumentParser(
        prog="psycho-rev",
        description="Analyze and reverse psycholinguistic difficulty features in LLM text.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="Compute all six psycholinguistic features for input text")
    p_an.add_argument("text", nargs="?", default=None, help="Text to analyze, or '-' for stdin")
    p_an.add_argument("--file", dest="input_file", default=None, help="Path to input file, or '-' for stdin")
    p_an.add_argument("--json", dest="as_json", action="store_true", help="Emit full JSON feature report")

    p_rw = sub.add_parser("rewrite", help="Rewrite text using a classical, neural, API, or hybrid reverser")
    p_rw.add_argument("text", nargs="?", default=None, help="Text to rewrite, or '-' for stdin")
    p_rw.add_argument("--file", dest="input_file", default=None, help="Path to input file, or '-' for stdin")
    p_rw.add_argument(
        "--method",
        choices=list(REWRITE_METHODS),
        default="classical",
        help="Rewriting backend to use (default: classical)",
    )
    p_rw.add_argument("--model", default=None, help="Model name for API or local neural backends")
    p_rw.add_argument("--base-url", default=None, help="Custom base URL for OpenAI-compatible API backends")
    p_rw.add_argument(
        "--adapter-dir",
        default=DEFAULT_LORA_ADAPTER,
        help="LoRA adapter directory or Hugging Face repo ID",
    )
    p_rw.add_argument("--samples", type=int, default=6, help="Candidate count for best-of-n / hybrid-bon")
    p_rw.add_argument(
        "--in-place",
        "-i",
        action="store_true",
        help="Edit --file in place when changes are made",
    )
    p_rw.add_argument("--diff", action="store_true", help="Print a unified diff of changes")
    p_rw.add_argument(
        "--plain",
        action="store_true",
        help="Emit only the rewritten text (suitable for Unix pipes)",
    )
    p_rw.add_argument("--json", dest="as_json", action="store_true", help="Emit full RewriteResult JSON")
    p_rw.add_argument(
        "--no-skip-clear",
        action="store_true",
        help="Disable clear-text gating and always rewrite",
    )
    p_rw.add_argument(
        "--unsafe-classical",
        action="store_true",
        help="Disable safe mode for the classical reverser",
    )

    p_srv = sub.add_parser("serve", help="Run the local HTTP JSON server")
    p_srv.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_srv.add_argument("--port", type=int, default=8787, help="Bind port (default: 8787)")
    p_srv.add_argument(
        "--method",
        choices=list(REWRITE_METHODS),
        default="classical",
        help="Rewriting backend to serve (default: classical)",
    )
    p_srv.add_argument("--model", default=None, help="Model name for API or local neural backends")
    p_srv.add_argument("--base-url", default=None, help="Custom base URL for OpenAI-compatible API backends")
    p_srv.add_argument(
        "--adapter-dir",
        default=DEFAULT_LORA_ADAPTER,
        help="LoRA adapter directory or Hugging Face repo ID",
    )
    p_srv.add_argument("--no-skip-clear", action="store_true", help="Disable clear-text gating")
    p_srv.add_argument("--unsafe-classical", action="store_true", help="Disable safe mode for classical backend")

    p_corpus = sub.add_parser("build-corpus", help="Build training/evaluation corpus")
    p_corpus.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments forwarded to corpus_builder")

    p_eval = sub.add_parser("evaluate", help="Run benchmark evaluation")
    p_eval.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments forwarded to evaluate")

    return parser


def _delegate_entrypoint(module_name: str, extra_args: list[str]) -> int:
    import importlib

    mod = importlib.import_module(module_name)
    main_fn = getattr(mod, "main")
    prev_argv = sys.argv
    sys.argv = [module_name, *extra_args]
    try:
        ret = main_fn()
    finally:
        sys.argv = prev_argv
    return int(ret) if isinstance(ret, int) else 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "build-corpus":
        return _delegate_entrypoint("psycho_rev.corpus_builder", raw_argv[1:])
    if raw_argv and raw_argv[0] == "evaluate":
        return _delegate_entrypoint("psycho_rev.evaluate", raw_argv[1:])

    parser = _build_parser()
    args = parser.parse_args(raw_argv)

    if args.cmd == "analyze":
        text, _ = _read_input_text(parser, args.text, args.input_file)
        from psycho_rev.rewriter import Rewriter

        rewriter = Rewriter(backend="classical")
        report = rewriter.analyze(text)
        data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        if args.as_json:
            print(json.dumps(data, indent=2))
        else:
            print(_format_analysis_text(data))
        return 0

    if args.cmd == "rewrite":
        if args.in_place and (not args.input_file or args.input_file == "-"):
            parser.error("--in-place requires --file <path> pointing to a real file.")

        from psycho_rev.rewriter import Rewriter, unified_diff

        need_metrics = bool(args.as_json)
        rewriter = Rewriter(
            backend=args.method,
            model=args.model,
            base_url=args.base_url,
            adapter_dir=args.adapter_dir,
            samples=args.samples,
            skip_if_clear=not args.no_skip_clear,
            safe_mode=not args.unsafe_classical,
            compute_metrics=need_metrics,
        )
        try:
            if args.in_place:
                result = rewriter.rewrite_file(args.input_file, in_place=True)
                filename = args.input_file
            else:
                text, filename = _read_input_text(parser, args.text, args.input_file)
                result = rewriter.rewrite(text)
        finally:
            rewriter.close()

        if args.as_json:
            print(json.dumps(result.to_dict(), indent=2))
        elif args.diff:
            diff_text = unified_diff(result.original, result.rewritten, filename=filename)
            if diff_text:
                sys.stdout.write(diff_text)
        elif args.plain or not args.in_place:
            sys.stdout.write(result.rewritten)
            if result.rewritten and not result.rewritten.endswith("\n"):
                sys.stdout.write("\n")
        return 0

    if args.cmd == "serve":
        from psycho_rev.rewriter import Rewriter
        from psycho_rev.server import serve

        rewriter = Rewriter(
            backend=args.method,
            model=args.model,
            base_url=args.base_url,
            adapter_dir=args.adapter_dir,
            skip_if_clear=not args.no_skip_clear,
            safe_mode=not args.unsafe_classical,
        )
        serve(rewriter, host=args.host, port=args.port)
        return 0

    if args.cmd == "build-corpus":
        return _delegate_entrypoint("psycho_rev.corpus_builder", list(args.extra_args))

    if args.cmd == "evaluate":
        return _delegate_entrypoint("psycho_rev.evaluate", list(args.extra_args))

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
