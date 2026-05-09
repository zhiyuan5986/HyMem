#!/usr/bin/env python3
"""Consolidate HyMem memory_exports with SimpleMem-compatible args/flow into llm_spans LanceDB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hymem.core.retriever import LanceDBLLMSpanRetriever, LanceDBMemorySummaryRetriever

# SIMPLEMEM_ROOT = REPO_ROOT / "hymem" / "ref" / "SimpleMem"
# if str(SIMPLEMEM_ROOT) not in sys.path:
#     sys.path.insert(0, str(SIMPLEMEM_ROOT))

from hymem.ref.SimpleMem.scripts.filter_extraction_trace_longllmlingua import TopKPPLPromptCompressor
from hymem.ref.SimpleMem.scripts.filter_extraction_trace_longllmlingua_laquer import align_entry_with_laquer, normalize_spans
from hymem.ref.SimpleMem.src.consts import LFQA_TASK
from hymem.ref.SimpleMem.src.laquer_methods.llm_method import LLMBasedAlignment  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate HyMem memory_exports to llm_spans (SimpleMem-compatible args).")
    parser.add_argument("--logs-dir", type=Path, required=True, help="Directory containing HyMem memory_exports sample_*.json files.")
    parser.add_argument("--db-search-roots", type=Path, nargs="+", default=[Path(".")], help="Roots to search sample DB dirs.")
    parser.add_argument("--output-suffix", type=str, default="longllmlingua_filtered")

    parser.add_argument("--compressor-model-name", type=str, required=True)
    parser.add_argument("--compressor-device-map", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--turn-window-k", type=int, default=2)
    parser.add_argument("--turn-separator", type=str, default="\n")
    parser.add_argument("--first-stage-filter", choices=["coarse_topk_by_ppl", "fine_topk_by_contrastive_ppl"], default="coarse_topk_by_ppl")
    parser.add_argument("--condition-in-question", choices=["none", "before", "after"], default="after")
    parser.add_argument("--condition-text", type=str, default="")
    parser.add_argument("--condition-placement", choices=["none", "prepend", "append"], default="prepend")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-trace-items", type=int, default=-1)
    parser.add_argument("--max-entries-per-item", type=int, default=-1)
    parser.add_argument("--fallback-spans-root", type=Path, default=Path("outputs/locomo10_spans_fallback"))
    return parser.parse_args()


def split_session_text(session_text: str) -> list[str]:
    return [line.strip() for line in session_text.split("\n") if line.strip()]


def find_memory_exports(logs_dir: Path) -> list[Path]:
    return sorted(logs_dir.glob("sample_*.json"), key=lambda p: int(p.stem.split("_")[-1]))


def find_sample_db_dir(sample_idx: int, roots: list[Path]) -> Path | None:
    target = f"lancedb_sample_{sample_idx}"
    for root in roots:
        root = root.resolve()
        direct = root / target
        if direct.exists() and direct.is_dir():
            return direct
        for candidate in root.glob(f"**/{target}"):
            if candidate.is_dir():
                return candidate
    return None


def main() -> None:
    args = parse_args()
    args.logs_dir = args.logs_dir.resolve()
    args.db_search_roots = [p.resolve() for p in args.db_search_roots]
    args.fallback_spans_root = args.fallback_spans_root.resolve()

    exports = find_memory_exports(args.logs_dir)
    if not exports:
        raise FileNotFoundError(f"No memory exports found under: {args.logs_dir}")

    compressor = TopKPPLPromptCompressor(model_name=args.compressor_model_name, device_map=args.compressor_device_map)
    aligner = LLMBasedAlignment(task=LFQA_TASK, args=args)

    summaries: list[dict[str, Any]] = []
    for export_file in exports:
        payload = json.loads(export_file.read_text(encoding="utf-8"))
        sample_idx = int(payload.get("sample_idx"))
        db_dir = find_sample_db_dir(sample_idx, args.db_search_roots)
        used_fallback_db = False
        if db_dir is None:
            db_dir = args.fallback_spans_root / f"lancedb_sample_{sample_idx}"
            used_fallback_db = True
        db_dir.mkdir(parents=True, exist_ok=True)

        memory_retriever = LanceDBMemorySummaryRetriever(db_path=str(db_dir))
        span_retriever = LanceDBLLMSpanRetriever(db_path=str(db_dir), table_name="llm_spans")

        pairs = payload.get("summary_note_pairs", [])
        max_items = len(pairs) if args.max_trace_items < 0 else min(args.max_trace_items, len(pairs))
        processed = 0
        for item_idx in range(max_items):
            pair = pairs[item_idx]
            summary = pair.get("summary") or {}
            note = pair.get("source_memory_note") or {}
            summary_id = summary.get("id")
            if not summary_id:
                continue
            if memory_retriever.get_entry_by_id(summary_id) is None:
                continue

            session_turns = split_session_text(note.get("content", ""))
            if not session_turns:
                continue

            entry_text = summary.get("content", "")
            coarse = compressor.compress_with_coarse_topk(
                context=[str(t) for t in session_turns],
                entry_text=entry_text,
                top_k=args.top_k,
                condition_in_question=args.condition_in_question,
                condition_text=args.condition_text,
                condition_placement=args.condition_placement,
                turn_window_k=args.turn_window_k,
                turn_separator=args.turn_separator,
                entry_budget_multiplier=1.0,
                first_stage_filter=args.first_stage_filter,
                ppl_plot_dir=Path("/tmp") / f"sample_{sample_idx}_{args.output_suffix}" / "ppl_plot",
                ppl_plot_file_stem=f"trace_{item_idx:04d}",
            )

            merged = coarse.get("coarse_merged_window", {})
            support_turns = []
            if merged:
                start_idx = int(merged.get("window_start", 0))
                end_idx = int(merged.get("window_end", start_idx))
                for idx in range(start_idx, min(end_idx + 1, len(session_turns))):
                    support_turns.append({"turn_index": idx, "text": session_turns[idx]})

            align_result = align_entry_with_laquer(aligner=aligner, entry_text=entry_text, context_turns=support_turns) if support_turns else {}
            raw_rows = align_result["results"].to_dict("records") if align_result and "results" in align_result else []
            spans = normalize_spans(rows=raw_rows, context_turns=support_turns)
            span_text = " ".join([s.get("span_text", "") for s in spans if s.get("span_text")]).strip()

            span_retriever.add_documents([{
                "id": summary_id,
                "content": span_text,
                "timestamp": summary.get("timestamp", ""),
                "metadata": {
                    "sample_idx": sample_idx,
                    "trace_item_index": item_idx,
                    "entry_id": summary_id,
                    "entry_text": entry_text,
                    "support_turns": support_turns,
                    "llm_spans": spans,
                    "llm_raw_spans": raw_rows,
                    "llm_response": {k: v for k, v in align_result.items() if k != "results"} if align_result else {},
                },
            }])
            processed += 1

        out_file = export_file.with_name(f"sample_{sample_idx}_{args.output_suffix}.json")
        out_file.write_text(json.dumps({"sample_idx": sample_idx, "processed": processed, "db_dir": str(db_dir)}, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append({
            "sample_idx": sample_idx,
            "memory_export": str(export_file),
            "output_json": str(out_file),
            "sample_db_dir": str(db_dir),
            "used_fallback_db": used_fallback_db,
            "trace_items_processed": processed,
        })

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
