
if __package__:
    from .project_paths import BASE_MODEL_DIR
else:
    from project_paths import BASE_MODEL_DIR

import json
import hashlib
import os
import random
import re
from pathlib import Path
from statistics import mean

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "data" / "multihoprag" / "multihoprag_eval.json"
VECTOR_DB = ROOT / "vector_db" / "multihoprag"
RESULTS_DIR = ROOT / "results"
ENV_FILE = ROOT / "config" / ".env"


COLLECTION_NAME = "multihoprag_bge_collection"
COLLECTION_NAME = "multihoprag_bge_collection_no_metadata"
INITIAL_TOP_K = 20
OUTPUT_TOP_K = 5
SAMPLE_SIZE = int(os.environ.get("MULTIHOPRAG_SAMPLE_SIZE", "5"))
LISTWISE_MODEL = "deepseek-v4-flash"
SETSELECTION_MODEL = "deepseek-v4-pro"


def client() -> OpenAI:
    load_dotenv(ENV_FILE)
    return OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


def json_escape(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().replace("\\", "\\\\").replace('"', "'")


def retrieve(collection, model, question: str):
    emb = model.encode(
        "Represent this sentence for searching relevant passages: " + question,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    result = collection.query(query_embeddings=[emb], n_results=INITIAL_TOP_K, include=["documents", "distances"])
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


def response_text(resp) -> str:
    message = resp.choices[0].message
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


def reasoning_text(resp) -> str:
    message = resp.choices[0].message
    return (getattr(message, "reasoning_content", "") or "").strip()


def response_usage(resp) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {}


def shuffled_candidates(question: str, ids: list[str], chunks: list[str]):
    seed = int(hashlib.md5(question.encode("utf-8")).hexdigest()[:8], 16)
    order = list(range(len(ids)))
    random.Random(seed).shuffle(order)
    return [ids[i] for i in order], [chunks[i] for i in order], order


def listwise_rerank(api: OpenAI, question: str, ids: list[str], chunks: list[str]):
    shuffled_ids, shuffled_chunks, shuffle_order = shuffled_candidates(question, ids, chunks)
    entries = [f"[{i}]\n{c.strip()}" for i, c in enumerate(shuffled_chunks)]
    prompt = f"""Do not explain. Do not analyze. Do not think step by step.
Rank the candidate passages by usefulness for answering the question.
Return exactly {OUTPUT_TOP_K} passage ids, ordered from most useful to least useful.
Output only comma-separated integers, for example: 3,1,7,12,0

Question:
{question}

Candidate passages:
{chr(10).join(entries)}

Ranking:"""
    resp = api.chat.completions.create(
        model=LISTWISE_MODEL,
        messages=[
            {"role": "system", "content": "You are a retrieval reranker. Return the final comma-separated passage ids immediately, with no explanation."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw_output = response_text(resp)
    raw_reasoning = reasoning_text(resp)
    order, parse_ok, parse_error = parse_order(raw_output, len(shuffled_chunks))
    return [shuffled_ids[i] for i in order], [shuffled_chunks[i] for i in order], raw_output, raw_reasoning, order, shuffle_order, parse_ok, parse_error, resp.choices[0].finish_reason, response_usage(resp)


def setselection(api: OpenAI, question: str, ids: list[str], chunks: list[str]):
    shuffled_ids, shuffled_chunks, shuffle_order = shuffled_candidates(question, ids, chunks)
    entries = [f"[{i}]\n{c.strip()}" for i, c in enumerate(shuffled_chunks)]
    prompt = f"""Do not explain. Do not analyze. Do not think step by step.
Select the {OUTPUT_TOP_K} passages that jointly provide the strongest evidence for answering the question.
The question may need multiple supporting passages.
Return exactly {OUTPUT_TOP_K} passage ids, ordered from strongest to weakest evidence.
Output only comma-separated integers, for example: 3,1,7,12,0

Question:
{question}

Candidate passages:
{chr(10).join(entries)}

Selected ids:"""
    resp = api.chat.completions.create(
        model=SETSELECTION_MODEL,
        messages=[
            {"role": "system", "content": "You are a retrieval selector. Return the final comma-separated passage ids immediately, with no explanation."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw_output = response_text(resp)
    raw_reasoning = reasoning_text(resp)
    order, parse_ok, parse_error = parse_order(raw_output, len(shuffled_chunks))
    return [shuffled_ids[i] for i in order], [shuffled_chunks[i] for i in order], raw_output, raw_reasoning, order, shuffle_order, parse_ok, parse_error, resp.choices[0].finish_reason, response_usage(resp)


def metrics(rows: list[dict]) -> dict:
    hit5 = []
    all5 = []
    rr = []
    for row in rows:
        gt = set(row["ground_truth_chunks"])
        ids = row["ids"]
        hit5.append(int(bool(gt & set(ids[:5]))))
        all5.append(int(gt.issubset(set(ids[:5]))))
        reciprocal = 0.0
        for idx, cid in enumerate(ids):
            if cid in gt:
                reciprocal = 1 / (idx + 1)
                break
        rr.append(reciprocal)
    return {"count": len(rows), "hit_at_5": mean(hit5), "all_support_at_5": mean(all5), "mrr": mean(rr)}


def main():
    items = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["test"][:SAMPLE_SIZE]
    embedding_model = SentenceTransformer(str(BASE_MODEL_DIR))
    collection = chromadb.PersistentClient(path=str(VECTOR_DB)).get_collection(name=COLLECTION_NAME)
    api = client()

    outputs = {"baseline_top20_first5": [], "listwise_top20_to5": [], "setselection_top20_to5": []}
    for item in tqdm(items, desc="sample"):
        ids, chunks = retrieve(collection, embedding_model, item["question"])
        outputs["baseline_top20_first5"].append({
            "id": item["id"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "ground_truth_chunks": item["ground_truth_chunks"],
            "ids": ids[:OUTPUT_TOP_K],
            "retrieved_chunks": chunks[:OUTPUT_TOP_K],
        })
        lw_ids, lw_chunks, lw_raw, lw_reasoning, lw_order, lw_shuffle, lw_parse_ok, lw_parse_error, lw_finish_reason, lw_usage = listwise_rerank(api, item["question"], ids, chunks)
        outputs["listwise_top20_to5"].append({
            "id": item["id"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "ground_truth_chunks": item["ground_truth_chunks"],
            "ids": lw_ids,
            "retrieved_chunks": lw_chunks,
            "raw_output": lw_raw,
            "raw_reasoning": lw_reasoning,
            "parsed_candidate_order": lw_order,
            "shuffle_order_original_indices": lw_shuffle,
            "parse_ok": lw_parse_ok,
            "parse_error": lw_parse_error,
            "finish_reason": lw_finish_reason,
            "usage": lw_usage,
        })
        ss_ids, ss_chunks, ss_raw, ss_reasoning, ss_order, ss_shuffle, ss_parse_ok, ss_parse_error, ss_finish_reason, ss_usage = setselection(api, item["question"], ids, chunks)
        outputs["setselection_top20_to5"].append({
            "id": item["id"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "ground_truth_chunks": item["ground_truth_chunks"],
            "ids": ss_ids,
            "retrieved_chunks": ss_chunks,
            "raw_output": ss_raw,
            "raw_reasoning": ss_reasoning,
            "parsed_candidate_order": ss_order,
            "shuffle_order_original_indices": ss_shuffle,
            "parse_ok": ss_parse_ok,
            "parse_error": ss_parse_error,
            "finish_reason": ss_finish_reason,
            "usage": ss_usage,
        })

    report = {name: metrics(rows) for name, rows in outputs.items()}
    payload = {"config": {
        "sample_size": SAMPLE_SIZE,
        "initial_top_k": INITIAL_TOP_K,
        "output_top_k": OUTPUT_TOP_K,
        "snippet_chars": None,
        "candidate_order": "deterministically shuffled before LLM prompt",
        "collection_name": COLLECTION_NAME,
        "listwise_model": LISTWISE_MODEL,
        "setselection_model": SETSELECTION_MODEL,
    }, "report": report, "items": outputs}
    out = RESULTS_DIR / f"multihoprag_llm_top20_sample{SAMPLE_SIZE}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
