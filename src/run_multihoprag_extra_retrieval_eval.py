import json
import hashlib
import os
import random
import re
from pathlib import Path
from statistics import mean

import chromadb
import numpy as np
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "data" / "multihoprag" / "multihoprag_eval.json"
VECTOR_DB = ROOT / "vector_db" / "multihoprag"
RESULTS_DIR = ROOT / "results"
ENV_FILE = ROOT / "config" / ".env"

BASE_MODEL_DIR = ROOT / "models" / "BAAI" / "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181"
COLLECTION_NAME = "multihoprag_bge_collection"

RETRIEVE_TOP_K = 50
METRIC_TOP_K = 20
GEN_TOP_K = 5
FLASH_MODEL = "deepseek-v4-flash"
SETSELECTION_MODEL = "deepseek-v4-pro"
LLM_SNIPPET_CHARS = 800


def ensure_client() -> OpenAI:
    load_dotenv(ENV_FILE)
    return OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )


def retrieve(collection, model: SentenceTransformer, question: str):
    instruction = "Represent this sentence for searching relevant passages: "
    query_embedding = model.encode(
        instruction + question,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=RETRIEVE_TOP_K,
        include=["documents", "distances"],
    )
    return result["ids"][0], result["documents"][0]


def parse_ranked_numbers(text: str, total: int, limit: int) -> list[int]:
    numbers = [int(x) for x in re.findall(r"\d+", text or "")]
    order = []
    seen = set()
    for num in numbers:
        if 0 <= num < total and num not in seen:
            order.append(num)
            seen.add(num)
        if len(order) == limit:
            break
    return order


def fill_order(order: list[int], total: int, limit: int) -> list[int]:
    seen = set(order)
    for idx in range(total):
        if idx not in seen:
            order.append(idx)
            seen.add(idx)
        if len(order) == limit:
            break
    return order[:limit]


def compact_chunk(chunk: str, max_chars: int = LLM_SNIPPET_CHARS) -> str:
    chunk = re.sub(r"\s+", " ", chunk).strip()
    if len(chunk) <= max_chars:
        return chunk
    return chunk[:max_chars].rstrip() + " ..."


def parse_json_numbers(text: str, key: str, total: int, limit: int) -> list[int]:
    try:
        data = json.loads(text)
        values = data.get(key, [])
        order = []
        seen = set()
        for value in values:
            idx = int(value)
            if 0 <= idx < total and idx not in seen:
                order.append(idx)
                seen.add(idx)
            if len(order) == limit:
                break
        return order
    except Exception:
        return parse_ranked_numbers(text, total, limit)


def shuffled_candidates(question: str, ids: list[str], chunks: list[str]) -> tuple[list[int], list[str], list[str]]:
    seed = int(hashlib.md5(question.encode("utf-8")).hexdigest()[:8], 16)
    order = list(range(len(chunks)))
    random.Random(seed).shuffle(order)
    return order, [ids[i] for i in order], [chunks[i] for i in order]


def colbert_rerank(model: BGEM3FlagModel, question: str, ids: list[str], chunks: list[str]) -> tuple[list[str], list[str]]:
    q_reps = model.encode([question], return_colbert_vecs=True)
    q_vec = np.asarray(q_reps["colbert_vecs"][0])

    d_reps = model.encode(chunks, batch_size=8, return_colbert_vecs=True)
    scores = []
    for d_vec in d_reps["colbert_vecs"]:
        d_vec = np.asarray(d_vec)
        sim = q_vec @ d_vec.T
        scores.append(float(sim.max(axis=1).sum() / q_vec.shape[0]))

    order = np.argsort(scores)[::-1].tolist()
    return [ids[i] for i in order], [chunks[i] for i in order]


def listwise_rerank(client: OpenAI, question: str, ids: list[str], chunks: list[str]) -> tuple[list[str], list[str]]:
    shuffle_order, shuffled_ids, shuffled_chunks = shuffled_candidates(question, ids, chunks)
    entries = []
    for idx, chunk in enumerate(shuffled_chunks):
        entries.append(f'{{"id": {idx}, "passage": "{compact_chunk(chunk).replace(chr(34), chr(39))}"}}')
    prompt = f"""You are a search ranking expert.

Rank the following {len(chunks)} candidate passages by how useful they are for answering the question.
Return all passage numbers in descending relevance order.
Output a JSON object only, in this exact shape:
{{"ranking":[3, 0, 12, ...]}}

Question:
{question}

Candidate passages:
[
{','.join(entries)}
]

JSON output:"""

    try:
        response = client.chat.completions.create(
            model=FLASH_MODEL,
            messages=[
                {"role": "system", "content": "You only output valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=256,
            stream=False,
        )
        raw = response.choices[0].message.content
        order = parse_json_numbers(raw, "ranking", len(shuffled_chunks), len(shuffled_chunks))
    except Exception as exc:
        print(f"Listwise failed: {exc}")
        order = []
    order = fill_order(order, len(shuffled_chunks), len(shuffled_chunks))
    return [shuffled_ids[i] for i in order], [shuffled_chunks[i] for i in order]


def setselection_rerank(client: OpenAI, question: str, ids: list[str], chunks: list[str]) -> tuple[list[str], list[str]]:
    shuffle_order, shuffled_ids, shuffled_chunks = shuffled_candidates(question, ids, chunks)
    entries = []
    for idx, chunk in enumerate(shuffled_chunks):
        entries.append(f'{{"id": {idx}, "passage": "{compact_chunk(chunk).replace(chr(34), chr(39))}"}}')
    prompt = f"""You are an evidence selection expert.

Select the {METRIC_TOP_K} candidate passages that are most useful for answering the question.
The question may require multiple pieces of evidence, so prefer a set that jointly covers all necessary facts.
Return exactly {METRIC_TOP_K} passage numbers ordered from most useful to least useful.
Output a JSON object only, in this exact shape:
{{"selected":[3, 0, 12, ...]}}

Question:
{question}

Candidate passages:
[
{','.join(entries)}
]

JSON output:"""

    try:
        response = client.chat.completions.create(
            model=SETSELECTION_MODEL,
            messages=[
                {"role": "system", "content": "You only output valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=128,
            stream=False,
        )
        raw = response.choices[0].message.content
        order = parse_json_numbers(raw, "selected", len(shuffled_chunks), METRIC_TOP_K)
    except Exception as exc:
        print(f"SetSelection failed: {exc}")
        order = []
    order = fill_order(order, len(shuffled_chunks), METRIC_TOP_K)
    return [shuffled_ids[i] for i in order], [shuffled_chunks[i] for i in order]


def metrics(rows: list[dict]) -> dict:
    hit1 = []
    hit5 = []
    all_support5 = []
    hit20 = []
    all_support20 = []
    rr = []
    for row in rows:
        gt = set(row["ground_truth_chunks"])
        ranked_ids = row["ids"]
        hit1.append(int(bool(ranked_ids[:1] and gt.intersection(ranked_ids[:1]))))
        hit5.append(int(bool(gt.intersection(ranked_ids[:5]))))
        all_support5.append(int(gt.issubset(set(ranked_ids[:5]))))
        hit20.append(int(bool(gt.intersection(ranked_ids[:20]))))
        all_support20.append(int(gt.issubset(set(ranked_ids[:20]))))
        reciprocal_rank = 0.0
        for idx, chunk_id in enumerate(ranked_ids):
            if chunk_id in gt:
                reciprocal_rank = 1.0 / (idx + 1)
                break
        rr.append(reciprocal_rank)
    return {
        "count": len(rows),
        "hit_at_1": mean(hit1),
        "hit_at_5": mean(hit5),
        "all_support_at_5": mean(all_support5),
        "hit_at_20": mean(hit20),
        "all_support_at_20": mean(all_support20),
        "mrr": mean(rr),
    }


def build_row(item: dict, ranked_ids: list[str], ranked_chunks: list[str]) -> dict:
    return {
        "id": item["id"],
        "question": item["question"],
        "gold_answer": item["gold_answer"],
        "ground_truth_chunks": item["ground_truth_chunks"],
        "ids": ranked_ids[:METRIC_TOP_K],
        "retrieved_chunks": ranked_chunks[:GEN_TOP_K],
    }


def save_rows(label: str, rows: list[dict]) -> None:
    path = RESULTS_DIR / f"multihoprag_{label}_retrieval.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    test_items = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["test"]

    print("加载 BGE-M3 embedding 模型并读取 Chroma...")
    embedding_model = SentenceTransformer(str(BASE_MODEL_DIR))
    collection = chromadb.PersistentClient(path=str(VECTOR_DB)).get_collection(name=COLLECTION_NAME)

    retrieved_cache = {}
    for item in tqdm(test_items, desc="initial retrieve top50"):
        retrieved_cache[item["id"]] = retrieve(collection, embedding_model, item["question"])

    report = {
        "dataset": "MultiHop-RAG-Mini-50",
        "initial_retrieve_top_k": RETRIEVE_TOP_K,
        "metric_top_k": METRIC_TOP_K,
        "saved_generation_top_k": GEN_TOP_K,
        "listwise_model": FLASH_MODEL,
        "setselection_model": SETSELECTION_MODEL,
        "llm_snippet_chars": LLM_SNIPPET_CHARS,
        "llm_candidate_order": "deterministically shuffled before prompting",
    }

    print("运行 ColBERT 重排...")
    colbert_model = BGEM3FlagModel(str(BASE_MODEL_DIR), use_fp16=True)
    colbert_rows = []
    for item in tqdm(test_items, desc="colbert"):
        ids, chunks = retrieved_cache[item["id"]]
        ranked_ids, ranked_chunks = colbert_rerank(colbert_model, item["question"], ids, chunks)
        colbert_rows.append(build_row(item, ranked_ids, ranked_chunks))
        save_rows("colbert", colbert_rows)
    report["colbert"] = metrics(colbert_rows)

    print("运行 DeepSeek Listwise 重排...")
    client = ensure_client()
    listwise_rows = []
    listwise_path = RESULTS_DIR / "multihoprag_listwise_retrieval.json"
    if listwise_path.exists():
        try:
            listwise_rows = json.loads(listwise_path.read_text(encoding="utf-8"))
        except Exception:
            listwise_rows = []
    done_ids = {row["id"] for row in listwise_rows}
    for item in tqdm(test_items, desc="listwise"):
        if item["id"] in done_ids:
            continue
        ids, chunks = retrieved_cache[item["id"]]
        ranked_ids, ranked_chunks = listwise_rerank(client, item["question"], ids, chunks)
        listwise_rows.append(build_row(item, ranked_ids, ranked_chunks))
        save_rows("listwise", listwise_rows)
    listwise_rows = sorted(listwise_rows, key=lambda row: row["id"])
    save_rows("listwise", listwise_rows)
    report["listwise"] = metrics(listwise_rows)

    print("运行 DeepSeek SetSelection...")
    setselection_rows = []
    setselection_path = RESULTS_DIR / "multihoprag_setselection_retrieval.json"
    if setselection_path.exists():
        try:
            setselection_rows = json.loads(setselection_path.read_text(encoding="utf-8"))
        except Exception:
            setselection_rows = []
    done_ids = {row["id"] for row in setselection_rows}
    for item in tqdm(test_items, desc="setselection"):
        if item["id"] in done_ids:
            continue
        ids, chunks = retrieved_cache[item["id"]]
        ranked_ids, ranked_chunks = setselection_rerank(client, item["question"], ids, chunks)
        setselection_rows.append(build_row(item, ranked_ids, ranked_chunks))
        save_rows("setselection", setselection_rows)
    setselection_rows = sorted(setselection_rows, key=lambda row: row["id"])
    save_rows("setselection", setselection_rows)
    report["setselection"] = metrics(setselection_rows)

    report_path = RESULTS_DIR / "multihoprag_extra_retrieval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
