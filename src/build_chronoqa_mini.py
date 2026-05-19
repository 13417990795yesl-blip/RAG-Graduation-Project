import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "chronoqa"
RAW_FILE = OUT_DIR / "chronoqa_source.json"
KB_FILE = OUT_DIR / "chronoqa_kb.md"
EVAL_FILE = OUT_DIR / "chronoqa_eval.json"
META_FILE = OUT_DIR / "chronoqa_mini_meta.json"

QUESTION_COUNT = 50
TARGET_CHUNKS = 1200
RANDOM_SEED = 20260518
CHUNKS_PER_QUESTION = TARGET_CHUNKS // QUESTION_COUNT


def normalize_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\uE000-\uF8FF]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_item(item: dict) -> bool:
    if not item.get("question") or not item.get("answer"):
        return False
    chunks = item.get("golden_chunks") or []
    return any(normalize_text(str(chunk)) for chunk in chunks)


def item_bucket(item: dict) -> tuple[str, str]:
    return (
        str(item.get("reference_document_count") or "unknown"),
        str(item.get("temporal_type") or "unknown"),
    )


def balanced_sample(items: list[dict]) -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    valid_items = [item for item in items if is_valid_item(item)]
    grouped = defaultdict(list)
    for item in valid_items:
        grouped[item_bucket(item)].append(item)

    # Keep ChronoQA-Mini deliberately harder than a random 50-question subset:
    # more multi-document cases, while still retaining absolute/relative coverage.
    quotas = {
        ("multiple", "aggregate"): 18,
        ("multiple", "absolute"): 8,
        ("multiple", "relative"): 6,
        ("single", "absolute"): 8,
        ("single", "relative"): 5,
        ("single", "aggregate"): 5,
    }

    selected = []
    selected_ids = set()
    for bucket, quota in quotas.items():
        candidates = grouped.get(bucket, [])
        rng.shuffle(candidates)
        for item in candidates[:quota]:
            selected.append(item)
            selected_ids.add(id(item))

    if len(selected) < QUESTION_COUNT:
        remaining = [item for item in valid_items if id(item) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: QUESTION_COUNT - len(selected)])

    return selected[:QUESTION_COUNT]


def positive_texts(item: dict) -> list[str]:
    texts = []
    seen = set()
    for chunk in item.get("golden_chunks") or []:
        text = normalize_text(str(chunk))
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def extract_time_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d{4}年|\d{1,2}月\d{1,2}日|\d{1,2}月|\d{4}-\d{2}-\d{2}", text or ""))


def char_bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", "", text or "")
    return {cleaned[i : i + 2] for i in range(max(0, len(cleaned) - 1))}


def build_distractor_candidates(items: list[dict], selected: list[dict]) -> list[dict]:
    selected_questions = {item["question"] for item in selected}
    candidates = []
    seen = set()
    for item in items:
        if item.get("question") in selected_questions:
            continue
        for chunk in item.get("golden_chunks") or []:
            text = normalize_text(str(chunk))
            if len(text) < 20 or text in seen:
                continue
            seen.add(text)
            candidates.append(
                {
                    "text": text,
                    "time_tokens": extract_time_tokens(text),
                    "bigrams": char_bigrams(text),
                    "reference_document_count": item.get("reference_document_count"),
                    "temporal_type": item.get("temporal_type"),
                    "answer_type": item.get("answer_type"),
                }
            )
    return candidates


def select_distractors(item: dict, candidates: list[dict], need: int, used_texts: set[str]) -> list[str]:
    q_time_tokens = (
        extract_time_tokens(item.get("question", ""))
        | extract_time_tokens(item.get("question_date", ""))
        | extract_time_tokens(item.get("event_time", "") or "")
    )
    q_bigrams = char_bigrams(item.get("question", ""))
    rng = random.Random(RANDOM_SEED + abs(hash(item.get("question", ""))) % 100000)
    scored = []
    for cand in candidates:
        text = cand["text"]
        if text in used_texts:
            continue
        score = 0
        score += len(cand["time_tokens"] & q_time_tokens) * 120
        score += len(cand["bigrams"] & q_bigrams)
        score += 20 if cand["reference_document_count"] == item.get("reference_document_count") else 0
        score += 20 if cand["temporal_type"] == item.get("temporal_type") else 0
        score += 10 if cand["answer_type"] == item.get("answer_type") else 0
        scored.append((score, rng.random(), text))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [text for _, _, text in scored[:need]]


def build_question_records(item: dict, qid: int, distractor_candidates: list[dict]) -> tuple[list[dict], list[str]]:
    positives = positive_texts(item)
    used = set(positives)
    distractors = select_distractors(
        item,
        distractor_candidates,
        max(0, CHUNKS_PER_QUESTION - len(positives)),
        used,
    )
    entries = [{"text": text, "source": "positive"} for text in positives]
    entries.extend({"text": text, "source": "distractor"} for text in distractors)

    rng = random.Random(RANDOM_SEED + qid)
    rng.shuffle(entries)

    records = []
    gt_ids = []
    for idx, entry in enumerate(entries[:CHUNKS_PER_QUESTION]):
        cid = f"题_{qid}_块_{idx}"
        records.append({"id": cid, "text": entry["text"], "source": entry["source"], "question_id": qid})
        if entry["source"] == "positive":
            gt_ids.append(cid)
    return records, gt_ids


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing raw ChronoQA file: {RAW_FILE}")

    items = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    selected = balanced_sample(items)

    distractor_candidates = build_distractor_candidates(items, selected)
    records = []
    eval_items = []
    for new_id, item in enumerate(selected):
        q_records, gt_ids = build_question_records(item, new_id, distractor_candidates)
        records.extend(q_records)
        eval_items.append(
            {
                "id": new_id,
                "question": item["question"],
                "gold_answer": item["answer"],
                "chunk_count": len(q_records),
                "ground_truth_chunks": gt_ids,
                "question_date": item.get("question_date"),
                "temporal_expression_type": item.get("temporal_expression_type"),
                "temporal_scope": item.get("temporal_scope"),
                "temporal_granularity": item.get("temporal_granularity"),
                "temporal_type": item.get("temporal_type"),
                "answer_type": item.get("answer_type"),
                "reference_document_count": item.get("reference_document_count"),
                "ref_prompt": item.get("ref_prompt"),
                "reference_qa_list": item.get("reference_qa_list"),
                "original_question": item.get("original_question"),
                "event_name": item.get("event_name"),
                "event_time": item.get("event_time"),
                "golden_chunks_urls": item.get("golden_chunks_urls") or [],
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    KB_FILE.write_text(
        "\n\n---\n\n".join(f"[{record['id']}] {record['text']}" for record in records) + "\n",
        encoding="utf-8",
    )
    EVAL_FILE.write_text(json.dumps({"test": eval_items}, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "source": "czy1999/ChronoQA chronoqa.json",
        "random_seed": RANDOM_SEED,
        "question_count": len(eval_items),
        "target_chunks": TARGET_CHUNKS,
        "chunks_per_question": CHUNKS_PER_QUESTION,
        "actual_chunks": len(records),
        "positive_chunks": sum(1 for record in records if record["source"] == "positive"),
        "distractor_chunks": sum(1 for record in records if record["source"] == "distractor"),
        "question_distribution": {
            "reference_document_count": Counter(item["reference_document_count"] for item in eval_items),
            "temporal_type": Counter(item["temporal_type"] for item in eval_items),
            "temporal_expression_type": Counter(item["temporal_expression_type"] for item in eval_items),
        },
        "files": {
            "kb": str(KB_FILE.relative_to(PROJECT_ROOT)),
            "eval": str(EVAL_FILE.relative_to(PROJECT_ROOT)),
        },
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
