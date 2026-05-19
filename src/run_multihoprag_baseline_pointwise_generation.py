from __future__ import annotations

import json
import os
import argparse
import re
from pathlib import Path
from statistics import mean

import chromadb
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import generate_multihoprag_english as generation
import run_multihoprag_llm_resume_eval as llm_eval


ROOT = Path(__file__).resolve().parents[1]
KB_FILE = ROOT / "data" / "multihoprag" / "multihoprag_kb.md"
EVAL_FILE = ROOT / "data" / "multihoprag" / "multihoprag_eval.json"
VECTOR_DB = ROOT / "vector_db" / "multihoprag"
RESULTS_DIR = ROOT / "results" / "multihoprag"

BASE_MODEL_DIR = ROOT / "models" / "BAAI" / "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_DIR = ROOT / "models" / "BAAI" / "models--BAAI--bge-reranker-v2-m3" / "snapshots" / "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

COLLECTION_NAME = "multihoprag_bge_collection_no_metadata"
SAMPLE_SIZE = int(os.environ.get("MULTIHOPRAG_SAMPLE_SIZE", "50"))
INITIAL_TOP_K = 20
OUTPUT_TOP_K = 5
METHODS = ("baseline", "pointwise", "colbert", "colbert_direct")


def parse_kb(path: Path = KB_FILE) -> tuple[list[str], list[str]]:
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


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def retrieve(collection, model: SentenceTransformer, question: str) -> tuple[list[str], list[str], list[float]]:
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
    ids = result["ids"][0]
    chunks = result["documents"][0]
    scores = [1.0 - distance for distance in result["distances"][0]]
    return ids, chunks, scores


def pointwise_rerank(
    reranker: FlagReranker,
    question: str,
    ids: list[str],
    chunks: list[str],
) -> tuple[list[str], list[str], list[float]]:
    scores = reranker.compute_score([[question, chunk] for chunk in chunks])
    if isinstance(scores, float):
        scores = [scores]
    ranked = sorted(zip(ids, chunks, scores), key=lambda item: item[2], reverse=True)
    return [item[0] for item in ranked], [item[1] for item in ranked], [float(item[2]) for item in ranked]


def colbert_rerank(
    model: BGEM3FlagModel,
    question: str,
    ids: list[str],
    chunks: list[str],
) -> tuple[list[str], list[str], list[float]]:
    q_reps = model.encode([question], return_colbert_vecs=True)
    q_vec = np.asarray(q_reps["colbert_vecs"][0])

    d_reps = model.encode(chunks, batch_size=8, return_colbert_vecs=True)
    scores = []
    for d_vec in d_reps["colbert_vecs"]:
        d_vec = np.asarray(d_vec)
        sim = q_vec @ d_vec.T
        scores.append(float(sim.max(axis=1).sum() / q_vec.shape[0]))

    ranked = sorted(zip(ids, chunks, scores), key=lambda item: item[2], reverse=True)
    return [item[0] for item in ranked], [item[1] for item in ranked], [float(item[2]) for item in ranked]


def encode_colbert_corpus(model: BGEM3FlagModel, chunks: list[str], batch_size: int = 8) -> list[np.ndarray]:
    vecs = []
    for start in tqdm(range(0, len(chunks), batch_size), desc="encode full KB colbert"):
        batch = chunks[start : start + batch_size]
        reps = model.encode(batch, batch_size=batch_size, return_colbert_vecs=True)
        vecs.extend(np.asarray(vec, dtype=np.float16) for vec in reps["colbert_vecs"])
    return vecs


def colbert_direct_search(
    model: BGEM3FlagModel,
    question: str,
    corpus_ids: list[str],
    corpus_chunks: list[str],
    corpus_vecs: list[np.ndarray],
    score_batch_size: int = 64,
) -> tuple[list[str], list[str], list[float]]:
    q_reps = model.encode([question], return_colbert_vecs=True)
    q_vec = torch.as_tensor(np.asarray(q_reps["colbert_vecs"][0], dtype=np.float16), device="cuda")

    scores = []
    with torch.no_grad():
        for start in range(0, len(corpus_vecs), score_batch_size):
            batch_vecs = corpus_vecs[start : start + score_batch_size]
            max_len = max(vec.shape[0] for vec in batch_vecs)
            dim = batch_vecs[0].shape[1]
            docs = torch.zeros((len(batch_vecs), max_len, dim), dtype=torch.float16, device="cuda")
            mask = torch.zeros((len(batch_vecs), max_len), dtype=torch.bool, device="cuda")
            for idx, vec in enumerate(batch_vecs):
                doc = torch.as_tensor(vec, dtype=torch.float16, device="cuda")
                docs[idx, : doc.shape[0]] = doc
                mask[idx, : doc.shape[0]] = True

            sim = torch.einsum("qd,bld->bql", q_vec, docs)
            sim = sim.masked_fill(~mask[:, None, :], torch.finfo(sim.dtype).min)
            batch_scores = sim.max(dim=2).values.sum(dim=1) / q_vec.shape[0]
            scores.extend(float(score) for score in batch_scores.detach().cpu())

    order = np.argsort(scores)[::-1][:INITIAL_TOP_K].tolist()
    return [corpus_ids[i] for i in order], [corpus_chunks[i] for i in order], [scores[i] for i in order]


def retrieval_metrics(rows: list[dict]) -> dict:
    hit1 = []
    hit5 = []
    all_support5 = []
    rr = []
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
    return {
        "count": len(rows),
        "hit_at_1": mean(hit1),
        "hit_at_5": mean(hit5),
        "all_support_at_5": mean(all_support5),
        "mrr": mean(rr),
    }


def retrieval_path(method: str) -> Path:
    return RESULTS_DIR / f"{method}_retrieval_top20_to5.json"


def generation_path(method: str) -> Path:
    return RESULTS_DIR / f"{method}_generation_top20_to5.json"


def generation_eval_path(method: str) -> Path:
    return RESULTS_DIR / f"{method}_generation_top20_to5_eval.json"


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def save_retrieval_report(method_rows: dict[str, list[dict]], sample_size: int) -> None:
    report_path = RESULTS_DIR / "llm_retrieval_top20_to5_report.json"
    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {}
    report.setdefault("dataset", "MultiHop-RAG")
    report["sample_size"] = sample_size
    report["initial_top_k"] = INITIAL_TOP_K
    report["output_top_k"] = OUTPUT_TOP_K
    report["collection_name"] = COLLECTION_NAME
    report["baseline_model"] = "bge-m3"
    report["pointwise_model"] = "bge-reranker-v2-m3"
    report["colbert_model"] = "bge-m3-colbert"
    report["colbert_direct_model"] = "bge-m3-colbert-full-kb"
    methods = report.setdefault("methods", {})
    for method, rows in method_rows.items():
        methods[method] = retrieval_metrics(rows)
    atomic_write_json(report_path, report)


def build_row(item: dict, initial_ids: list[str], ids: list[str], chunks: list[str], scores: list[float] | None = None) -> dict:
    row = {
        "id": item["id"],
        "question": item["question"],
        "gold_answer": item["gold_answer"],
        "ground_truth_chunks": item["ground_truth_chunks"],
        "initial_top20_ids": initial_ids,
        "ids": ids[:OUTPUT_TOP_K],
        "retrieved_chunks": chunks[:OUTPUT_TOP_K],
    }
    if scores is not None:
        row["scores"] = scores[:OUTPUT_TOP_K]
    return row


def run_retrieval(sample_size: int, methods: tuple[str, ...], write_files: bool = True) -> dict[str, list[dict]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    items = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["test"][:sample_size]

    needs_dense_retrieval = bool(set(methods) - {"colbert_direct"})
    if needs_dense_retrieval:
        print("Loading BGE-M3 embedding model on CUDA-visible device...")
    embedding_model = SentenceTransformer(str(BASE_MODEL_DIR), device="cuda") if needs_dense_retrieval else None
    collection = (
        chromadb.PersistentClient(path=str(VECTOR_DB)).get_collection(name=COLLECTION_NAME)
        if needs_dense_retrieval
        else None
    )

    retrieved_cache = {}
    baseline_rows = []
    if needs_dense_retrieval:
        for item in tqdm(items, desc="baseline retrieval"):
            ids, chunks, scores = retrieve(collection, embedding_model, item["question"])
            retrieved_cache[item["id"]] = (ids, chunks)
            if "baseline" in methods:
                baseline_rows.append(build_row(item, ids, ids, chunks, scores))
                if write_files:
                    atomic_write_json(retrieval_path("baseline"), baseline_rows)

    method_rows = {}
    if "baseline" in methods:
        method_rows["baseline"] = baseline_rows

    if "pointwise" in methods:
        print("Loading BGE reranker on CUDA-visible device...")
        reranker = FlagReranker(str(RERANKER_DIR), use_fp16=True, devices=["cuda:0"])
        pointwise_rows = []
        for item in tqdm(items, desc="pointwise rerank"):
            ids, chunks = retrieved_cache[item["id"]]
            ranked_ids, ranked_chunks, rerank_scores = pointwise_rerank(reranker, item["question"], ids, chunks)
            pointwise_rows.append(build_row(item, ids, ranked_ids, ranked_chunks, rerank_scores))
            if write_files:
                atomic_write_json(retrieval_path("pointwise"), pointwise_rows)
        method_rows["pointwise"] = pointwise_rows

    if "colbert" in methods:
        print("Loading BGE-M3 ColBERT model on CUDA-visible device...")
        colbert_model = BGEM3FlagModel(str(BASE_MODEL_DIR), use_fp16=True, devices=["cuda:0"])
        colbert_rows = []
        for item in tqdm(items, desc="colbert rerank"):
            ids, chunks = retrieved_cache[item["id"]]
            ranked_ids, ranked_chunks, colbert_scores = colbert_rerank(colbert_model, item["question"], ids, chunks)
            colbert_rows.append(build_row(item, ids, ranked_ids, ranked_chunks, colbert_scores))
            if write_files:
                atomic_write_json(retrieval_path("colbert"), colbert_rows)
        method_rows["colbert"] = colbert_rows

    if "colbert_direct" in methods:
        print("Loading BGE-M3 ColBERT model for direct full-KB search on CUDA-visible device...")
        colbert_direct_model = BGEM3FlagModel(str(BASE_MODEL_DIR), use_fp16=True, devices=["cuda:0"])
        corpus_ids, corpus_chunks = parse_kb()
        corpus_vecs = encode_colbert_corpus(colbert_direct_model, corpus_chunks)
        colbert_direct_rows = []
        for item in tqdm(items, desc="colbert direct search"):
            ranked_ids, ranked_chunks, colbert_scores = colbert_direct_search(
                colbert_direct_model,
                item["question"],
                corpus_ids,
                corpus_chunks,
                corpus_vecs,
            )
            colbert_direct_rows.append(build_row(item, ranked_ids, ranked_ids, ranked_chunks, colbert_scores))
            if write_files:
                atomic_write_json(retrieval_path("colbert_direct"), colbert_direct_rows)
        method_rows["colbert_direct"] = colbert_direct_rows

    if write_files:
        save_retrieval_report(method_rows, sample_size)
    return method_rows


def run_generation_and_update_report(methods: tuple[str, ...]) -> dict[str, dict]:
    summaries = {}
    for method in methods:
        rows = generation.run_generation(method, retrieval_path(method), generation_path(method))
        summaries[method] = generation.evaluate(method, rows, generation_eval_path(method))

    report_path = RESULTS_DIR / "llm_retrieval_top20_to5_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["generation_model"] = generation.GENERATION_MODEL
    report["generation_top_k"] = generation.GENERATION_TOP_K
    report["generation_max_tokens"] = None
    generation_methods = report.setdefault("generation_methods", {})
    generation_methods.update(summaries)
    atomic_write_json(report_path, report)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MultiHop-RAG baseline/pointwise/colbert retrieval and generation.")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--smoke-test", action="store_true", help="Run retrieval and one generation per method without writing result files.")
    return parser.parse_args()


def run_smoke_test(sample_size: int, methods: tuple[str, ...]) -> None:
    method_rows = run_retrieval(sample_size, methods, write_files=False)
    client = generation.make_client()
    output = {}
    for method, rows in method_rows.items():
        row = rows[0]
        result = generation.generate_answer(client, row["question"], row["retrieved_chunks"][: generation.GENERATION_TOP_K])
        precision, recall, f1, em = generation.calculate_item_metrics(result["answer"], row["gold_answer"])
        output[method] = {
            "retrieval": retrieval_metrics(rows),
            "sample_answer": result["answer"],
            "finish_reason": result["finish_reason"],
            "chunks": len(row["retrieved_chunks"]),
            "item_metrics": {"precision": precision, "recall": recall, "f1": f1, "em": em},
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    if args.smoke_test:
        run_smoke_test(args.sample_size, methods)
        return

    method_rows = run_retrieval(args.sample_size, methods)
    summaries = run_generation_and_update_report(methods)
    report = json.loads((RESULTS_DIR / "llm_retrieval_top20_to5_report.json").read_text(encoding="utf-8"))
    print(json.dumps({"retrieval": {k: retrieval_metrics(v) for k, v in method_rows.items()}, "generation": summaries, "report": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
