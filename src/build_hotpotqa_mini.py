import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "hotpotqa"
KB_FILE = OUT_DIR / "hotpotqa_kb.md"
EVAL_FILE = OUT_DIR / "hotpotqa_eval.json"
SOURCE_FILE = OUT_DIR / "hotpotqa_validation_50_source.json"
META_FILE = OUT_DIR / "hotpotqa_mini_meta.json"

QUESTION_COUNT = 50
RANDOM_SEED = 20260518


def clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def paragraph_text(title: str, sentences: list[str]) -> str:
    body = clean_text(" ".join(clean_text(sentence) for sentence in sentences))
    return f"{title}. {body}" if title else body


def is_good_item(item: dict) -> bool:
    titles = item["context"]["title"]
    sentences = item["context"]["sentences"]
    sf_titles = set(item["supporting_facts"]["title"])
    if len(titles) < 8:
        return False
    if not item.get("answer") or item["answer"].strip().lower() in {"", "unknown"}:
        return False
    return len(sf_titles.intersection(titles)) >= 1


def build_one(item: dict, qid: int) -> tuple[list[dict], dict]:
    titles = item["context"]["title"]
    paragraphs = item["context"]["sentences"]
    sf_pairs = set(zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"]))
    sf_titles = set(item["supporting_facts"]["title"])

    records = []
    gt_ids = []
    supporting_facts_by_chunk = {}
    for idx, (title, sentences) in enumerate(zip(titles, paragraphs)):
        cid = f"题_{qid}_块_{idx}"
        text = paragraph_text(title, sentences)
        records.append({"id": cid, "title": title, "text": text})

        sentence_ids = [sid for sf_title, sid in sf_pairs if sf_title == title]
        if title in sf_titles:
            gt_ids.append(cid)
            supporting_facts_by_chunk[cid] = sorted(sentence_ids)

    eval_item = {
        "id": qid,
        "source_id": item["id"],
        "question": item["question"],
        "gold_answer": item["answer"],
        "chunk_count": len(records),
        "ground_truth_chunks": gt_ids,
        "type": item.get("type"),
        "level": item.get("level"),
        "supporting_facts": {
            "title": item["supporting_facts"]["title"],
            "sent_id": item["supporting_facts"]["sent_id"],
        },
        "supporting_facts_by_chunk": supporting_facts_by_chunk,
    }
    return records, eval_item


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    candidates = [item for item in dataset if is_good_item(item)]

    hard = [item for item in candidates if item.get("level") == "hard"]
    medium = [item for item in candidates if item.get("level") == "medium"]
    easy = [item for item in candidates if item.get("level") == "easy"]
    rng.shuffle(hard)
    rng.shuffle(medium)
    rng.shuffle(easy)
    selected = hard[:QUESTION_COUNT]
    selected.extend(medium[: max(0, QUESTION_COUNT - len(selected))])
    selected.extend(easy[: max(0, QUESTION_COUNT - len(selected))])
    selected = selected[:QUESTION_COUNT]

    records = []
    eval_items = []
    source_items = []
    for qid, item in enumerate(selected):
        q_records, eval_item = build_one(item, qid)
        records.extend(q_records)
        eval_items.append(eval_item)
        source_items.append(item)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    KB_FILE.write_text(
        "\n\n---\n\n".join(f"[{record['id']}] {record['text']}" for record in records) + "\n",
        encoding="utf-8",
    )
    EVAL_FILE.write_text(json.dumps({"test": eval_items}, ensure_ascii=False, indent=2), encoding="utf-8")
    SOURCE_FILE.write_text(json.dumps(source_items, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "source": "hotpotqa/hotpot_qa distractor validation",
        "random_seed": RANDOM_SEED,
        "question_count": len(eval_items),
        "actual_chunks": len(records),
        "chunks_per_question_values": sorted(set(item["chunk_count"] for item in eval_items)),
        "ground_truth_chunk_count_distribution": Counter(len(item["ground_truth_chunks"]) for item in eval_items),
        "level_distribution": Counter(item["level"] for item in eval_items),
        "type_distribution": Counter(item["type"] for item in eval_items),
        "files": {
            "kb": str(KB_FILE.relative_to(ROOT)),
            "eval": str(EVAL_FILE.relative_to(ROOT)),
            "source": str(SOURCE_FILE.relative_to(ROOT)),
        },
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
