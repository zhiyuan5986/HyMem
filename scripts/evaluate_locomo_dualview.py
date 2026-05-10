import os, re, json, argparse, gc
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from hymem.agent import HybridMemAgent
from hymem.data.loader import load_locomo_dataset
from scripts.evaluate_locomo import setup_logger, process_conversation, load_cached_memories, save_cached_memories

RECALL_CATEGORIES = {1,2,3,4}

def _collect_ids_from_text(text: str):
    return set(re.findall(r"(session_\d+:[^,\s]+)", text or ""))

def _collect_ids_from_llm_spans(span_entry):
    ids = set()
    md = getattr(span_entry, "metadata", {}) or {}
    if not isinstance(md, dict):
        return ids

    for dia_key in ("support_turn_dia_ids", "evidence", "evidences"):
        v = md.get(dia_key)
        if isinstance(v, list):
            ids.update({str(x).strip() for x in v if str(x).strip()})

    llm_spans = md.get("llm_spans")
    if isinstance(llm_spans, list):
        for span in llm_spans:
            if not isinstance(span, dict):
                continue
            for k in ("dia_id", "dialogue_id", "legacy_dialogue_id"):
                v = span.get(k)
                if v is not None and str(v).strip():
                    ids.add(str(v).strip())
            ids.update(_collect_ids_from_text(span.get("span_text", "")))

    ids.update(_collect_ids_from_text(getattr(span_entry, "content", "")))
    return ids

def recall_from_indices(indices, qa_evidence, summary_entries, span_entries):
    gold = {str(x).strip() for x in (qa_evidence or []) if str(x).strip()}
    if not gold:
        return None, [], []

    span_entry_by_id = {}
    for entry in span_entries:
        entry_id = getattr(entry, "id", None)
        if entry_id:
            span_entry_by_id[entry_id] = entry

    pred = set()
    for idx in indices:
        if 0 <= idx < len(summary_entries):
            summary_entry = summary_entries[idx]
            span_entry = span_entry_by_id.get(getattr(summary_entry, "id", None))
            if span_entry is not None:
                pred.update(_collect_ids_from_llm_spans(span_entry))
    rec = len(pred & gold) / len(gold) if gold else None
    return rec, sorted(pred), sorted(gold)

def _collect_route_indices(debug_payload):
    return {
        "light": {
            "summary_semantic_idx": debug_payload.get("light_hybrid", {}).get("mem_sem_indices", []),
            "summary_keyword_idx": debug_payload.get("light_hybrid", {}).get("mem_lex_indices", []),
            "span_semantic_idx": debug_payload.get("light_hybrid", {}).get("span_sem_indices", []),
            "span_keyword_idx": debug_payload.get("light_hybrid", {}).get("span_lex_indices", []),
        },
        "deep": {
            "summary_semantic_idx": debug_payload.get("deep_hybrid", {}).get("mem_sem_indices", []),
            "summary_keyword_idx": debug_payload.get("deep_hybrid", {}).get("mem_lex_indices", []),
            "span_semantic_idx": debug_payload.get("deep_hybrid", {}).get("span_sem_indices", []),
            "span_keyword_idx": debug_payload.get("deep_hybrid", {}).get("span_lex_indices", []),
        }
    }

def _materialize_route_entries(agent, route_indices):
    def _from_summary(indices):
        items = []
        for idx in indices:
            if 0 <= idx < len(agent.memory_system.retriever.entries):
                entry = agent.memory_system.retriever.entries[idx]
                items.append({
                    "idx": idx,
                    "id": entry.id,
                    "link": entry.link,
                    "timestamp": entry.timestamp,
                    "content": entry.content,
                    "metadata": entry.metadata,
                })
        return items

    def _from_span(indices):
        items = []
        for idx in indices:
            if 0 <= idx < len(agent.memory_system.span_retriever.entries):
                entry = agent.memory_system.span_retriever.entries[idx]
                items.append({
                    "idx": idx,
                    "id": entry.id,
                    "timestamp": entry.timestamp,
                    "content": entry.content,
                    "metadata": entry.metadata,
                })
        return items

    return {
        "light": {
            "summary_semantic_entries": _from_summary(route_indices["light"]["summary_semantic_idx"]),
            "summary_keyword_entries": _from_summary(route_indices["light"]["summary_keyword_idx"]),
            "span_semantic_entries": _from_span(route_indices["light"]["span_semantic_idx"]),
            "span_keyword_entries": _from_span(route_indices["light"]["span_keyword_idx"]),
        },
        "deep": {
            "summary_semantic_entries": _from_summary(route_indices["deep"]["summary_semantic_idx"]),
            "summary_keyword_entries": _from_summary(route_indices["deep"]["summary_keyword_idx"]),
            "span_semantic_entries": _from_span(route_indices["deep"]["span_semantic_idx"]),
            "span_keyword_entries": _from_span(route_indices["deep"]["span_keyword_idx"]),
        },
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='./data/locomo10.json')
    p.add_argument('--cache_dir', type=str, default='./cache')
    p.add_argument('--output', type=str, default='./results/locomo_dualview_recall.json')
    p.add_argument('--ratio', type=float, default=1.0)
    p.add_argument('--backend', type=str, default='openai')
    p.add_argument('--temperature', type=float, default=0.5)
    p.add_argument('--model_name', type=str, default='gpt-4.1-mini')
    p.add_argument('--embed_model', type=str, default='text-embedding-3-small')
    p.add_argument('--api_key', type=str, default='')
    p.add_argument('--embed_api_key', type=str, default='')
    p.add_argument('--base_url', type=str, default='')
    p.add_argument('--embed_base_url', type=str, default='')
    p.add_argument('--retrieve_k', type=int, default=15)
    p.add_argument('--retrieve_k_rough', type=int, default=30)
    p.add_argument('--enable_hybrid', action='store_true', default=True)
    p.add_argument('--mem_sem_weight', type=float, default=0.65)
    p.add_argument('--mem_lex_weight', type=float, default=0.35)
    p.add_argument('--span_sem_weight', type=float, default=0.45)
    p.add_argument('--span_lex_weight', type=float, default=0.55)
    p.add_argument('--final_mem_weight', type=float, default=0.45)
    p.add_argument('--final_span_weight', type=float, default=0.45)
    p.add_argument('--final_agree_weight', type=float, default=0.10)
    p.add_argument('--rrf_k', type=float, default=60.0)
    p.add_argument('--mem_sem_k', type=int, default=15)
    p.add_argument('--mem_lex_k', type=int, default=15)
    p.add_argument('--span_sem_k', type=int, default=15)
    p.add_argument('--span_lex_k', type=int, default=15)
    args = p.parse_args()

    log_path = os.path.join(os.path.dirname(__file__), 'logs', f'locomo_dualview_{datetime.now().strftime("%Y%m%d%H%M%S")}.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path)

    samples = load_locomo_dataset(os.path.join(os.path.dirname(__file__), args.dataset))
    if args.ratio < 1.0:
        samples = samples[:max(1, int(len(samples)*args.ratio))]

    cache_dir = os.path.join(os.path.dirname(__file__), args.cache_dir)
    results = []
    recall_stats = defaultdict(list)

    for sample_idx, sample in enumerate(tqdm(samples, total=len(samples))):
        agent = HybridMemAgent(embed_model=args.embed_model, db_path=cache_dir+f"/lancedb_sample_{sample_idx}", model_name=args.model_name,
                               embed_api_key=args.embed_api_key, api_key=args.api_key, embed_base_url=args.embed_base_url,
                               base_url=args.base_url, backend=args.backend, retrieve_k=args.retrieve_k,
                               temperature=args.temperature, k_rough=args.retrieve_k_rough)
        agent.memory_system.enable_hybrid = args.enable_hybrid
        agent.memory_system.hybrid_score_weights.update({
            "mem_sem_weight": args.mem_sem_weight,
            "mem_lex_weight": args.mem_lex_weight,
            "span_sem_weight": args.span_sem_weight,
            "span_lex_weight": args.span_lex_weight,
            "final_mem_weight": args.final_mem_weight,
            "final_span_weight": args.final_span_weight,
            "final_agree_weight": args.final_agree_weight,
            "rrf_k": args.rrf_k,
        })
        agent.memory_system.hybrid_route_top_k.update({
            "mem_sem_k": args.mem_sem_k,
            "mem_lex_k": args.mem_lex_k,
            "span_sem_k": args.span_sem_k,
            "span_lex_k": args.span_lex_k,
        })

        if not load_cached_memories(agent, cache_dir, sample_idx, args.embed_model, args.embed_api_key, args.embed_base_url, logger):
            process_conversation(agent, sample, logger)
            save_cached_memories(agent, cache_dir, sample_idx, logger)

        for qa in sample.qa:
            if int(qa.category or 0) not in RECALL_CATEGORIES:
                continue
            pred, _ = agent.memory_system.dynamic_retrieval(qa.question, k=args.retrieve_k, k_rough=args.retrieve_k_rough)
            stats = agent.get_last_retrieval_stats()
            debug = stats.get('retrieval_debug', {})
            route_indices = _collect_route_indices(debug)
            route_entries = _materialize_route_entries(agent, route_indices)
            light_indices = debug.get('light_indices', [])
            deep_indices = debug.get('deep_indices', [])

            light_recall, light_pred_ids, gold_ids = recall_from_indices(
                light_indices,
                qa.evidence,
                agent.memory_system.summary_list,
                agent.memory_system.span_retriever.entries,
            )
            deep_recall, deep_pred_ids, _ = recall_from_indices(
                deep_indices,
                qa.evidence,
                agent.memory_system.summary_list,
                agent.memory_system.span_retriever.entries,
            )
            if light_recall is not None:
                recall_stats['light'].append(light_recall)
            if deep_recall is not None:
                recall_stats['deep'].append(deep_recall)
            print("light_pred_ids:",light_pred_ids)
            print(f"light_recall: {light_recall}, deep_recall: {deep_recall}")

            results.append({
                'sample_idx': sample_idx,
                'question': qa.question,
                'category': int(qa.category),
                'answer': pred,
                'evidence': qa.evidence,
                'gold_ids': gold_ids,
                'light_recall': light_recall,
                'deep_recall': deep_recall,
                'light_predicted_ids': light_pred_ids,
                'deep_predicted_ids': deep_pred_ids,
                'retrieval_debug': debug,
                'four_route_entry_indices': route_indices,
                'four_route_entries': route_entries,
                'dynamic_stats': stats,
            })

        del agent
        gc.collect()

    summary = {
        'num_records': len(results),
        'avg_light_recall': (sum(recall_stats['light'])/len(recall_stats['light'])) if recall_stats['light'] else None,
        'avg_deep_recall': (sum(recall_stats['deep'])/len(recall_stats['deep'])) if recall_stats['deep'] else None,
    }

    out = {'summary': summary, 'results': results}
    out_path = os.path.join(os.path.dirname(__file__), args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f'saved to {out_path}')

if __name__ == '__main__':
    main()
