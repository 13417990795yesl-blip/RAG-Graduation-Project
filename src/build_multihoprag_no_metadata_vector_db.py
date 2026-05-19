import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
KB_FILE = ROOT / "data" / "multihoprag" / "multihoprag_kb.md"
VECTOR_DB = ROOT / "vector_db" / "multihoprag"
BASE_MODEL_DIR = ROOT / "models" / "BAAI" / "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181"
COLLECTION_NAME = "multihoprag_bge_collection_no_metadata"


def parse_kb():
    ids = []
    chunks = []
    content = KB_FILE.read_text(encoding="utf-8")
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


def batches(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i : i + size]


def main():
    ids, chunks = parse_kb()
    model = SentenceTransformer(str(BASE_MODEL_DIR))
    client = chromadb.PersistentClient(path=str(VECTOR_DB))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() > 0:
        collection.delete(ids=collection.get(include=[])["ids"])
    for start, id_batch in tqdm(list(batches(ids, 96)), desc="write no-metadata multihoprag"):
        chunk_batch = chunks[start : start + len(id_batch)]
        embeddings = model.encode(
            chunk_batch,
            batch_size=12,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.add(ids=id_batch, documents=chunk_batch, embeddings=embeddings)
    print({"collection": COLLECTION_NAME, "count": collection.count()})


if __name__ == "__main__":
    main()
