#!/usr/bin/env python3
"""Coarse-to-fine extraction trace filtering with LAQuer-style LLM span mining.

This script keeps the coarse LongLLMLingua stage (turn-window PPL ranking) and
replaces token-level fine filtering with an LLM-based evidence span extraction
step inspired by LAQuer (ACL 2025).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from hymem.ref.SimpleMem.src.consts import HYMEM_TASK
from hymem.ref.SimpleMem.src.laquer_methods.llm_method import LLMBasedAlignment  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coarse LongLLMLingua + LAQuer-style LLM span extraction for *_extraction_trace.json"
    )
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)

    parser.add_argument("--compressor-model-name", type=str, required=True)
    parser.add_argument("--compressor-device-map", type=str, default="cuda")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--turn-window-k", type=int, default=2)
    parser.add_argument("--turn-separator", type=str, default="\n")
    parser.add_argument(
        "--first-stage-filter",
        choices=["coarse_topk_by_ppl", "fine_topk_by_contrastive_ppl"],
        default="coarse_topk_by_ppl",
        help="Document/window ranking strategy used in stage-1 selection.",
    )
    parser.add_argument("--condition-in-question", choices=["none", "before", "after"], default="after")
    parser.add_argument("--condition-text", type=str, default="")
    parser.add_argument("--condition-placement", choices=["none", "prepend", "append"], default="prepend")

    parser.add_argument("--model", type=str, required=True, help="Model identifier passed to LAQuer inference wrapper.")

    parser.add_argument("--max-trace-items", type=int, default=-1)
    parser.add_argument("--max-entries-per-item", type=int, default=-1)
    parser.add_argument("--min-span-confidence", type=float, default=0.0)
    parser.add_argument(
        "--spans-db-dir",
        type=Path,
        default=None,
        help="Directory for storing entry-aligned LLM span LanceDB (default: sibling folder of output-json).",
    )
    return parser.parse_args()


def normalize_spans(rows: list[dict[str, Any]], context_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_map = {f"turn_{turn['turn_index']}": turn for turn in context_turns}
    normalized: list[dict[str, Any]] = []

    def _parse_first_offset(raw_offsets: Any) -> tuple[int, int]:
        if not isinstance(raw_offsets, list) or len(raw_offsets) == 0:
            return -1, -1
        first = raw_offsets[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            try:
                return int(first[0]), int(first[1])
            except (TypeError, ValueError):
                return -1, -1
        return -1, -1

    for row in rows:
        # print(row)
        source_id = str(row.get("documentFile", ""))
        turn = source_map.get(source_id)
        if not turn:
            continue

        offset_start, offset_end = _parse_first_offset(row.get("docSpanOffsets", []))

        normalized.append(
            {
                "turn_index": turn["turn_index"],
                "speaker": turn["speaker"],
                "span_text": str(row.get("docSpanText", "")).strip(),
                "char_start": offset_start,
                "char_end": offset_end,
                "is_verbatim_match": offset_start >= 0,
            }
        )

    normalized.sort(key=lambda x: (x["turn_index"], x["char_start"]))
    return normalized


def align_entry_with_laquer(aligner: LLMBasedAlignment, entry_text: str, context_turns: list[dict[str, Any]]) -> dict[str, Any]:
    source_spans = {f"turn_{turn['turn_index']}": turn["text"] for turn in context_turns}
    source_metadata = {
        f"turn_{turn['turn_index']}": [
            {
                "documentFile": f"turn_{turn['turn_index']}",
                "docSpanText": turn["text"],
                "docSpanOffsets": [[0, len(turn["text"])]],
                "docSentCharIdx": 0,
                "docSentText": turn["text"],
            }
        ]
        for turn in context_turns
    }
    datapoint = {
        "topic": "locomo_trace",
        "unique_id": "locomo_trace",
        "source_spans": source_spans,
        "source_metadata": source_metadata,
        "sentence": entry_text,
        "scuSpanOffsets": [0, len(entry_text)],
        "complete_scuSentence": entry_text,
        "is_sampled": False,
        "source_granularity": "sentence",
        "fact_idx": 0,
        "question": entry_text,
    }
    return aligner.extract_attribution(datapoint)




__all__ = ["HYMEM_TASK", "LLMBasedAlignment", "align_entry_with_laquer", "normalize_spans"]
