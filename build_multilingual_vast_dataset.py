"""
build_multilingual_vast_dataset.py

Extracts a balanced, vast collection of questions & passages from ai4bharat/MSMARCO-XI
across all 12 Indic languages and English (200+ queries per language = 2,500+ questions total).
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "msmarco_subset.json")

LANGUAGE_PARQUETS = {
    "hi": "validation/hinval.parquet",
    "ta": "validation/tamval.parquet",
    "te": "validation/telval.parquet",
    "bn": "validation/benval.parquet",
    "kn": "validation/kanval.parquet",
    "ml": "validation/malval.parquet",
    "mr": "validation/marval.parquet",
    "gu": "validation/gujval.parquet",
    "pa": "validation/panval.parquet",
    "as": "validation/asmval.parquet",
    "or": "validation/orival.parquet",
    "ur": "validation/urdval.parquet"
}


def build_balanced_vast_dataset(queries_per_indic_lang: int = 200, max_total_passages: int = 8000) -> Dict[str, Any]:
    from data.download_dataset import generate_rich_indic_fallback
    fallback = generate_rich_indic_fallback(limit=200)
    
    all_passages = list(fallback["passages"])
    all_queries = list(fallback["queries"])
    
    seen_passages = set(p["text"].strip() for p in all_passages)
    seen_queries = set(q["query_text"].strip() for q in all_queries)
    
    doc_counter = len(all_passages) + 1
    query_counter = len(all_queries) + 1
    
    lang_query_counts = {}
    for q in all_queries:
        l = q.get("language", "hi")
        lang_query_counts[l] = lang_query_counts.get(l, 0) + 1

    logger.info(f"Extracting balanced vast MSMARCO-XI corpus ({queries_per_indic_lang} queries per language, max passages: {max_total_passages})...")
    
    for lang, parquet_path in LANGUAGE_PARQUETS.items():
        logger.info(f"Streaming language '{lang}' from {parquet_path}...")
        try:
            ds_stream = load_dataset(
                "ai4bharat/MSMARCO-XI",
                data_files={"validation": parquet_path},
                split="validation",
                streaming=True
            )
            
            lang_q_count = 0
            for idx, item in enumerate(ds_stream):
                if lang_q_count >= queries_per_indic_lang:
                    break
                
                indic_q = (item.get("query") or "").strip()
                eng_q = (item.get("Eng_Query") or "").strip()
                indic_ans = (item.get("Answer") or "").strip()
                eng_ans = (item.get("Eng_Answer") or "").strip()
                
                passages_dict = item.get("passages", {}) or {}
                translated_passages = passages_dict.get("Translated_passages", []) or []
                english_passages = passages_dict.get("English_passages", []) or []
                is_selected_list = passages_dict.get("is_selected", []) or []
                
                rel_docs_for_item = []
                
                # Process translated (Indic) passages
                for p_idx, p_text in enumerate(translated_passages):
                    if not p_text or not isinstance(p_text, str) or len(p_text.strip()) < 15:
                        continue
                    p_clean = p_text.strip()
                    is_sel = (p_idx < len(is_selected_list) and is_selected_list[p_idx] == 1)
                    
                    doc_id = f"doc_xi_{doc_counter:06d}"
                    doc_counter += 1
                    
                    if p_clean not in seen_passages and len(all_passages) < max_total_passages:
                        seen_passages.add(p_clean)
                        all_passages.append({
                            "doc_id": doc_id,
                            "text": p_clean,
                            "topic": f"msmarco_xi_{lang}",
                            "metadata": {
                                "source": "ai4bharat/MSMARCO-XI",
                                "language": lang,
                                "is_groundtruth": is_sel,
                                "groundtruth_answer": indic_ans or eng_ans or p_clean
                            }
                        })
                    
                    if is_sel:
                        rel_docs_for_item.append(doc_id)
                
                # Process English passages
                for p_idx, p_text in enumerate(english_passages):
                    if not p_text or not isinstance(p_text, str) or len(p_text.strip()) < 15:
                        continue
                    p_clean = p_text.strip()
                    is_sel = (p_idx < len(is_selected_list) and is_selected_list[p_idx] == 1)
                    
                    doc_id = f"doc_xi_{doc_counter:06d}"
                    doc_counter += 1
                    
                    if p_clean not in seen_passages and len(all_passages) < max_total_passages:
                        seen_passages.add(p_clean)
                        all_passages.append({
                            "doc_id": doc_id,
                            "text": p_clean,
                            "topic": "msmarco_xi_en",
                            "metadata": {
                                "source": "ai4bharat/MSMARCO-XI",
                                "language": "en",
                                "is_groundtruth": is_sel,
                                "groundtruth_answer": eng_ans or indic_ans or p_clean
                            }
                        })
                    
                    if is_sel:
                        rel_docs_for_item.append(doc_id)
                
                # Add Indic query
                if indic_q and indic_q not in seen_queries:
                    seen_queries.add(indic_q)
                    all_queries.append({
                        "query_id": f"q_xi_{query_counter:06d}",
                        "query_text": indic_q,
                        "relevant_doc_ids": rel_docs_for_item,
                        "language": lang,
                        "groundtruth_answer": indic_ans if indic_ans else (eng_ans or "Answer provided in passage.")
                    })
                    query_counter += 1
                    lang_q_count += 1
                    lang_query_counts[lang] = lang_query_counts.get(lang, 0) + 1
                
                # Add English query if balanced
                if eng_q and eng_q not in seen_queries and lang_query_counts.get("en", 0) < 600:
                    seen_queries.add(eng_q)
                    all_queries.append({
                        "query_id": f"q_xi_{query_counter:06d}_en",
                        "query_text": eng_q,
                        "relevant_doc_ids": rel_docs_for_item,
                        "language": "en",
                        "groundtruth_answer": eng_ans if eng_ans else (indic_ans or "Answer provided in passage.")
                    })
                    query_counter += 1
                    lang_query_counts["en"] = lang_query_counts.get("en", 0) + 1
                    
            logger.info(f"Language '{lang}' complete: extracted {lang_q_count} queries.")
            
        except Exception as e:
            logger.error(f"Error streaming language {lang}: {e}", exc_info=True)
            
    dataset_dict = {
        "passages": all_passages[:max_total_passages],
        "queries": all_queries,
        "metadata": {
            "dataset_name": "ai4bharat/MSMARCO-XI",
            "total_passages": len(all_passages[:max_total_passages]),
            "total_queries": len(all_queries),
            "source": "huggingface_ai4bharat_msmarco_xi_vast_multilingual"
        }
    }
    
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Successfully wrote vast dataset to {OUTPUT_FILE}: {len(all_queries)} queries, {len(all_passages[:max_total_passages])} passages.")
    logger.info(f"Final language distribution: {lang_query_counts}")
    return dataset_dict


if __name__ == "__main__":
    build_balanced_vast_dataset(queries_per_indic_lang=200, max_total_passages=8000)
