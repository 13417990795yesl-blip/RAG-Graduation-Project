
if __package__:
    from .project_paths import BASE_MODEL_DIR
else:
    from project_paths import BASE_MODEL_DIR

if __package__:
    from .data_utils import read_kb_chunks
else:
    from data_utils import read_kb_chunks

import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
KB_FILE = ROOT / "data" / "multihoprag" / "multihoprag_kb.md"
VECTOR_DB = ROOT / "vector_db" / "multihoprag"

COLLECTION_NAME = "multihoprag_bge_collection_no_metadata"


def parse_kb():
    return read_kb_chunks(KB_FILE)


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
