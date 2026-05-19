import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "multihoprag"
KB_FILE = OUT_DIR / "multihoprag_kb.md"
EVAL_FILE = OUT_DIR / "multihoprag_eval.json"
CORPUS_SOURCE_FILE = OUT_DIR / "corpus.json"
QA_SOURCE_FILE = OUT_DIR / "MultiHopRAG.json"
META_FILE = OUT_DIR / "multihoprag_mini_meta.json"

TMP_CORPUS = Path("/tmp/multihoprag_corpus.json")
TMP_QA = Path("/tmp/multihoprag_qa.json")

QUESTION_COUNT = 50
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
INCLUDE_METADATA_IN_KB = False


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = clean_text(text)
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    step = size - overlap
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks


def build_corpus(corpus: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    records = []
    url_to_chunks = {}
    for doc_idx, doc in enumerate(corpus):
        title = clean_text(doc.get("title", ""))
        body = clean_text(doc.get("body", ""))
        doc_chunks = []
        for chunk_idx, chunk in enumerate(chunk_text(body)):
            cid = f"题_{doc_idx}_块_{chunk_idx}"
            if INCLUDE_METADATA_IN_KB:
                metadata = " | ".join(
                    part
                    for part in [
                        f"Title: {title}" if title else "",
                        f"Source: {doc.get('source')}" if doc.get("source") else "",
                        f"Category: {doc.get('category')}" if doc.get("category") else "",
                        f"Published: {doc.get('published_at')}" if doc.get("published_at") else "",
                    ]
                    if part
                )
                text = f"{metadata}\n{chunk}" if metadata else chunk
            else:
                text = chunk
            record = {"id": cid, "text": text, "url": doc.get("url"), "title": title}
            records.append(record)
            doc_chunks.append(record)
        if doc.get("url"):
            url_to_chunks[doc["url"]] = doc_chunks
    return records, url_to_chunks


def find_fact_chunk(fact: str, chunks: list[dict]) -> str | None:
    fact_norm = clean_text(fact)
    if not fact_norm:
        return None
    for chunk in chunks:
        if fact_norm in clean_text(chunk["text"]):
            return chunk["id"]
    # Fallback for minor whitespace/punctuation differences.
    fact_tokens = set(re.findall(r"[A-Za-z0-9]+", fact_norm.lower()))
    if not fact_tokens:
        return None
    best_id = None
    best_overlap = 0
    for chunk in chunks:
        tokens = set(re.findall(r"[A-Za-z0-9]+", chunk["text"].lower()))
        overlap = len(fact_tokens & tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = chunk["id"]
    return best_id if best_overlap >= max(4, int(len(fact_tokens) * 0.6)) else None


def build_eval_items(qa_items: list[dict], url_to_chunks: dict[str, list[dict]]) -> list[dict]:
    eval_items = []
    for item in qa_items:
        gt_ids = []
        evidence_meta = []
        ok = True
        for ev in item.get("evidence_list", []):
            chunks = url_to_chunks.get(ev.get("url"))
            if not chunks:
                ok = False
                break
            chunk_id = find_fact_chunk(ev.get("fact", ""), chunks)
            if not chunk_id:
                ok = False
                break
            if chunk_id not in gt_ids:
                gt_ids.append(chunk_id)
            evidence_meta.append(
                {
                    "title": ev.get("title"),
                    "url": ev.get("url"),
                    "source": ev.get("source"),
                    "category": ev.get("category"),
                    "published_at": ev.get("published_at"),
                    "fact": ev.get("fact"),
                    "chunk_id": chunk_id,
                }
            )
        if not ok or len(gt_ids) < 2:
            continue
        eval_items.append(
            {
                "id": len(eval_items),
                "question": item["query"],
                "gold_answer": item["answer"],
                "question_type": item.get("question_type"),
                "ground_truth_chunks": gt_ids,
                "evidence_count": len(item.get("evidence_list", [])),
                "evidence": evidence_meta,
            }
        )
        if len(eval_items) == QUESTION_COUNT:
            break
    return eval_items


def main() -> None:
    corpus = json.loads(TMP_CORPUS.read_text(encoding="utf-8"))
    qa_items = json.loads(TMP_QA.read_text(encoding="utf-8"))

    records, url_to_chunks = build_corpus(corpus)
    eval_items = build_eval_items(qa_items, url_to_chunks)
    if len(eval_items) < QUESTION_COUNT:
        raise RuntimeError(f"Only built {len(eval_items)} evaluable questions")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_SOURCE_FILE.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    QA_SOURCE_FILE.write_text(json.dumps(qa_items, ensure_ascii=False, indent=2), encoding="utf-8")
    KB_FILE.write_text(
        "\n\n---\n\n".join(f"[{record['id']}] {record['text']}" for record in records) + "\n",
        encoding="utf-8",
    )
    EVAL_FILE.write_text(json.dumps({"test": eval_items}, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "source": "yixuantt/MultiHop-RAG dataset",
        "question_count": len(eval_items),
        "document_count": len(corpus),
        "actual_chunks": len(records),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "include_metadata_in_kb": INCLUDE_METADATA_IN_KB,
        "ground_truth_chunk_count_distribution": Counter(len(item["ground_truth_chunks"]) for item in eval_items),
        "question_type_distribution": Counter(item["question_type"] for item in eval_items),
        "files": {
            "kb": str(KB_FILE.relative_to(ROOT)),
            "eval": str(EVAL_FILE.relative_to(ROOT)),
            "corpus_source": str(CORPUS_SOURCE_FILE.relative_to(ROOT)),
            "qa_source": str(QA_SOURCE_FILE.relative_to(ROOT)),
        },
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
