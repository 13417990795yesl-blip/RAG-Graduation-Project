from __future__ import annotations

if __package__:
    from .project_paths import BASE_MODEL_DIR
else:
    from project_paths import BASE_MODEL_DIR


import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "data" / "multihoprag" / "multihoprag_eval.json"
VECTOR_DB = ROOT / "vector_db" / "multihoprag"
RESULTS_DIR = ROOT / "results" / "multihoprag"
ENV_FILE = ROOT / "config" / ".env"


COLLECTION_NAME = "multihoprag_bge_collection_no_metadata"
INITIAL_TOP_K = 20
OUTPUT_TOP_K = 5
DEFAULT_SAMPLE_SIZE = int(os.environ.get("MULTIHOPRAG_SAMPLE_SIZE", "50"))
DEFAULT_METHODS = [m.strip() for m in os.environ.get("MULTIHOPRAG_METHODS", "listwise,setselection").split(",") if m.strip()]
LISTWISE_MODEL = "deepseek-v4-flash"
SETSELECTION_MODEL = "deepseek-v4-pro"


def ensure_client():
    from openai import OpenAI

    load_dotenv(ENV_FILE)
    return OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def retrieve(collection, model, question: str) -> tuple[list[str], list[str]]:
    embedding = model.encode(
        "Represent this sentence for searching relevant passages: " + question,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    result = collection.query(
        query_embeddings=[embedding],
        n_results=INITIAL_TOP_K,
        include=["documents", "distances"],
    )
    return result["ids"][0], result["documents"][0]


def parse_order(raw: str, total: int) -> tuple[list[int], bool, str]:
    values = re.findall(r"\d+", raw or "")
    if not values:
        return [], False, "no numeric ids found in model output"

    order = []
    seen = set()
    for value in values:
        idx = int(value)
        if 0 <= idx < total and idx not in seen:
            order.append(idx)
            seen.add(idx)
        if len(order) == OUTPUT_TOP_K:
            break

    if not order:
        return [], False, "numeric ids were all out of candidate range"

    for idx in range(total):
        if idx not in seen:
            order.append(idx)
            seen.add(idx)
        if len(order) == OUTPUT_TOP_K:
            break
    return order[:OUTPUT_TOP_K], True, ""


def response_text(response) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(getattr(part, "text", "")))
        return "".join(parts).strip()
    return (content or "").strip()


def reasoning_text(response) -> str:
    message = response.choices[0].message
    return (getattr(message, "reasoning_content", "") or "").strip()


def response_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {}


def shuffled_candidates(question: str, ids: list[str], chunks: list[str]) -> tuple[list[str], list[str], list[int]]:
    seed = int(hashlib.md5(question.encode("utf-8")).hexdigest()[:8], 16)
    order = list(range(len(ids)))
    random.Random(seed).shuffle(order)
    return [ids[i] for i in order], [chunks[i] for i in order], order


def build_prompt(method: str, question: str, chunks: list[str]) -> str:
    entries = [f"[{idx}]\n{chunk.strip()}" for idx, chunk in enumerate(chunks)]
    if method == "listwise":
        task = "Rank the candidate passages by usefulness for answering the question."
        suffix = "Ranking:"
    elif method == "setselection":
        task = (
            f"Select the {OUTPUT_TOP_K} passages that jointly provide the strongest evidence for answering the question.\n"
            "The question may need multiple supporting passages."
        )
        suffix = "Selected ids:"
    else:
        raise ValueError(f"Unknown method: {method}")

    return f"""Do not explain. Do not analyze. Do not think step by step.
{task}
Return exactly {OUTPUT_TOP_K} passage ids, ordered from most useful to least useful.
Output only comma-separated integers, for example: 3,1,7,12,0

Question:
{question}

Candidate passages:
{chr(10).join(entries)}

{suffix}"""


def llm_rerank(api: OpenAI, method: str, question: str, ids: list[str], chunks: list[str]) -> dict:
    shuffled_ids, shuffled_chunks, shuffle_order = shuffled_candidates(question, ids, chunks)
    prompt = build_prompt(method, question, shuffled_chunks)
    model_name = LISTWISE_MODEL if method == "listwise" else SETSELECTION_MODEL
    system_message = (
        "You are a retrieval reranker. Return the final comma-separated passage ids immediately, with no explanation."
        if method == "listwise"
        else "You are a retrieval selector. Return the final comma-separated passage ids immediately, with no explanation."
    )
    response = api.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw_output = response_text(response)
    order, parse_ok, parse_error = parse_order(raw_output, len(shuffled_chunks))
    ranked_ids = [shuffled_ids[i] for i in order]
    ranked_chunks = [shuffled_chunks[i] for i in order]
    return {
        "ids": ranked_ids,
        "retrieved_chunks": ranked_chunks,
        "raw_output": raw_output,
        "raw_reasoning": reasoning_text(response),
        "parsed_candidate_order": order,
        "shuffle_order_original_indices": shuffle_order,
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "finish_reason": response.choices[0].finish_reason,
        "usage": response_usage(response),
    }


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}

    hit1 = []
    hit5 = []
    all_support5 = []
    rr = []
    parse_ok = []
    for row in rows:
        gt = set(row["ground_truth_chunks"])
        ranked_ids = row["ids"]
        hit1.append(int(bool(ranked_ids[:1] and gt.intersection(ranked_ids[:1]))))
        hit5.append(int(bool(gt.intersection(ranked_ids[:5]))))
        all_support5.append(int(gt.issubset(set(ranked_ids[:5]))))
        reciprocal_rank = 0.0
        for idx, chunk_id in enumerate(ranked_ids):
            if chunk_id in gt:
                reciprocal_rank = 1.0 / (idx + 1)
                break
        rr.append(reciprocal_rank)
        parse_ok.append(int(bool(row.get("parse_ok"))))

    return {
        "count": len(rows),
        "parse_ok_rate": mean(parse_ok),
        "hit_at_1": mean(hit1),
        "hit_at_5": mean(hit5),
        "all_support_at_5": mean(all_support5),
        "mrr": mean(rr),
    }


def result_path(method: str) -> Path:
    return RESULTS_DIR / f"{method}_retrieval_top20_to5.json"


def load_rows(method: str) -> list[dict]:
    path = result_path(method)
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, list):
            return rows
    except Exception:
        pass
    return []


def is_complete(row: dict) -> bool:
    return bool(row.get("parse_ok")) and len(row.get("ids", [])) >= OUTPUT_TOP_K


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumeable MultiHop-RAG LLM reranking evaluation for listwise and setselection.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of questions to evaluate from the beginning of the eval split. Default: env MULTIHOPRAG_SAMPLE_SIZE or 50.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods to run: listwise,setselection. Default: env MULTIHOPRAG_METHODS or both.",
    )
    return parser.parse_args()


def save_report(all_rows: dict[str, list[dict]], sample_size: int) -> None:
    report = {
        "dataset": "MultiHop-RAG",
        "sample_size": sample_size,
        "initial_top_k": INITIAL_TOP_K,
        "output_top_k": OUTPUT_TOP_K,
        "collection_name": COLLECTION_NAME,
        "candidate_order": "deterministically shuffled before LLM prompt",
        "snippet_chars": None,
        "listwise_model": LISTWISE_MODEL,
        "setselection_model": SETSELECTION_MODEL,
        "methods": {method: metrics(rows) for method, rows in all_rows.items()},
    }
    atomic_write_json(RESULTS_DIR / "llm_retrieval_top20_to5_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    args = parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown_methods = sorted(set(methods) - {"listwise", "setselection"})
    if unknown_methods:
        raise ValueError(f"Unknown methods: {', '.join(unknown_methods)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    items = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["test"][:args.sample_size]

    print("Loading BGE-M3 embedding model and no-metadata MultiHop-RAG collection...")
    embedding_model = SentenceTransformer(str(BASE_MODEL_DIR))
    collection = chromadb.PersistentClient(path=str(VECTOR_DB)).get_collection(name=COLLECTION_NAME)
    api = ensure_client()

    all_rows = {}
    for method in methods:
        rows = load_rows(method)
        rows_by_id = {row["id"]: row for row in rows if is_complete(row)}
        pending = [item for item in items if item["id"] not in rows_by_id]

        print(f"{method}: {len(rows_by_id)}/{len(items)} completed, {len(pending)} pending")
        for item in tqdm(pending, desc=method):
            initial_ids, initial_chunks = retrieve(collection, embedding_model, item["question"])
            try:
                reranked = llm_rerank(api, method, item["question"], initial_ids, initial_chunks)
            except Exception as exc:
                reranked = {
                    "ids": [],
                    "retrieved_chunks": [],
                    "raw_output": "",
                    "raw_reasoning": "",
                    "parsed_candidate_order": [],
                    "shuffle_order_original_indices": [],
                    "parse_ok": False,
                    "parse_error": f"api error: {exc}",
                    "finish_reason": "",
                    "usage": {},
                }

            row = {
                "id": item["id"],
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "ground_truth_chunks": item["ground_truth_chunks"],
                "initial_top20_ids": initial_ids,
                **reranked,
            }
            rows_by_id[item["id"]] = row
            atomic_write_json(result_path(method), [rows_by_id[i] for i in sorted(rows_by_id)])
            save_report({**all_rows, method: [rows_by_id[i] for i in sorted(rows_by_id)]}, args.sample_size)

        method_rows = [rows_by_id[i] for i in sorted(rows_by_id)]
        atomic_write_json(result_path(method), method_rows)
        all_rows[method] = method_rows

    save_report(all_rows, args.sample_size)


if __name__ == "__main__":
    main()
