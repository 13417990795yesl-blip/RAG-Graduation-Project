import json
import re
from pathlib import Path
from statistics import mean
from typing import Iterable

import chromadb
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
KB_FILE = ROOT / "data" / "hotpotqa" / "hotpotqa_kb.md"
EVAL_FILE = ROOT / "data" / "hotpotqa" / "hotpotqa_eval.json"
VECTOR_DB = ROOT / "vector_db" / "hotpotqa"
RESULTS_DIR = ROOT / "results"

BASE_MODEL_DIR = ROOT / "models" / "BAAI" / "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_DIR = ROOT / "models" / "BAAI" / "models--BAAI--bge-reranker-v2-m3" / "snapshots" / "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

COLLECTION_NAME = "hotpotqa_bge_collection"
RETRIEVE_TOP_K = 20
GEN_TOP_K = 2


def parse_kb(path: Path) -> tuple[list[str], list[str]]:
    content = path.read_text(encoding="utf-8")
    ids = []
    chunks = []
    for block in content.split("---"):
        block = block.strip()
        if not block:
            continue
        match = re.search(r"\[(题_\d+_块_\d+)\]", block)
        if not match:
            continue
        ids.append(match.group(1))
        chunks.append(re.sub(r"\[题_\d+_块_\d+\]", "", block).strip())
    return ids, chunks


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def ensure_collection(model: SentenceTransformer):
    VECTOR_DB.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    ids, chunks = parse_kb(KB_FILE)

    if collection.count() == len(ids):
        return collection

    if collection.count() > 0:
        collection.delete(ids=collection.get(include=[])["ids"])

    for id_batch in tqdm(list(batched(ids, 96)), desc="写入 ChronoQA Chroma"):
        start = ids.index(id_batch[0])
        chunk_batch = chunks[start : start + len(id_batch)]
        embeddings = model.encode(
            chunk_batch,
            batch_size=12,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.add(ids=id_batch, documents=chunk_batch, embeddings=embeddings)
    return collection


def retrieve(collection, model: SentenceTransformer, question: str, top_k: int = RETRIEVE_TOP_K):
    instruction = "Represent this sentence for searching relevant passages: "
    query_embedding = model.encode(
        instruction + question,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"],
    )
    ids = result["ids"][0]
    chunks = result["documents"][0]
    scores = [1.0 - distance for distance in result["distances"][0]]
    return ids, chunks, scores


def pointwise_rerank(reranker: FlagReranker, question: str, ids: list[str], chunks: list[str]):
    if not ids:
        return [], []
    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.compute_score(pairs)
    ranked = sorted(zip(ids, chunks, scores), key=lambda item: item[2], reverse=True)
    return [item[0] for item in ranked], [item[1] for item in ranked]


def metrics(rows: list[dict]) -> dict:
    hit1 = []
    hit2 = []
    all_support2 = []
    hit20 = []
    rr = []
    for row in rows:
        gt = set(row["ground_truth_chunks"])
        ranked_ids = row["ids"]
        hit1.append(int(bool(ranked_ids[:1] and gt.intersection(ranked_ids[:1]))))
        hit2.append(int(bool(gt.intersection(ranked_ids[:2]))))
        all_support2.append(int(gt.issubset(set(ranked_ids[:2]))))
        hit20.append(int(bool(gt.intersection(ranked_ids[:20]))))
        reciprocal_rank = 0.0
        for idx, chunk_id in enumerate(ranked_ids):
            if chunk_id in gt:
                reciprocal_rank = 1.0 / (idx + 1)
                break
        rr.append(reciprocal_rank)
    return {
        "count": len(rows),
        "hit_at_1": mean(hit1),
        "hit_at_2": mean(hit2),
        "all_support_at_2": mean(all_support2),
        "hit_at_20": mean(hit20),
        "mrr": mean(rr),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    test_items = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["test"]

    print("加载 BGE-M3 embedding 模型...")
    embedding_model = SentenceTransformer(str(BASE_MODEL_DIR))
    collection = ensure_collection(embedding_model)

    print("运行 baseline 检索...")
    baseline_rows = []
    retrieved_cache = {}
    for item in tqdm(test_items, desc="baseline"):
        ids, chunks, scores = retrieve(collection, embedding_model, item["question"])
        retrieved_cache[item["id"]] = (ids, chunks)
        baseline_rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "ground_truth_chunks": item["ground_truth_chunks"],
                "ids": ids,
                "scores": scores,
                "retrieved_chunks": chunks[:GEN_TOP_K],
            }
        )

    baseline_metrics = metrics(baseline_rows)
    (RESULTS_DIR / "hotpotqa_baseline_retrieval.json").write_text(
        json.dumps(baseline_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("加载 BGE reranker，运行 pointwise 重排...")
    reranker = FlagReranker(str(RERANKER_DIR), use_fp16=True)
    pointwise_rows = []
    for item in tqdm(test_items, desc="pointwise"):
        ids, chunks = retrieved_cache[item["id"]]
        ranked_ids, ranked_chunks = pointwise_rerank(reranker, item["question"], ids, chunks)
        pointwise_rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "ground_truth_chunks": item["ground_truth_chunks"],
                "ids": ranked_ids,
                "retrieved_chunks": ranked_chunks[:GEN_TOP_K],
            }
        )

    pointwise_metrics = metrics(pointwise_rows)
    (RESULTS_DIR / "hotpotqa_pointwise_retrieval.json").write_text(
        json.dumps(pointwise_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "dataset": "HotpotQA-Distractor-Mini-50",
        "kb_chunks": collection.count(),
        "retrieve_top_k": RETRIEVE_TOP_K,
        "saved_generation_top_k": GEN_TOP_K,
        "baseline": baseline_metrics,
        "pointwise": pointwise_metrics,
    }
    (RESULTS_DIR / "hotpotqa_retrieval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
