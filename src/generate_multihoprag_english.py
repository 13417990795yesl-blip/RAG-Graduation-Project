import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "multihoprag"
ENV_FILE = ROOT / "config" / ".env"

INPUT_FILES = {
    "listwise": RESULTS_DIR / "listwise_retrieval_top20_to5.json",
    "setselection": RESULTS_DIR / "setselection_retrieval_top20_to5.json",
}

OUTPUT_FILES = {
    "listwise": RESULTS_DIR / "listwise_generation_top20_to5.json",
    "setselection": RESULTS_DIR / "setselection_generation_top20_to5.json",
}

EVAL_FILES = {
    "listwise": RESULTS_DIR / "listwise_generation_top20_to5_eval.json",
    "setselection": RESULTS_DIR / "setselection_generation_top20_to5_eval.json",
}

GENERATION_MODEL = "deepseek-v4-flash"
GENERATION_TOP_K = 5
REPORT_FILE = RESULTS_DIR / "llm_retrieval_top20_to5_report.json"


def make_client() -> OpenAI:
    load_dotenv(ENV_FILE)
    return OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )


def response_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {}


def generate_answer(client: OpenAI, question: str, chunks: list[str]) -> dict:
    if not chunks:
        return {"answer": "I don't know.", "finish_reason": "no_context", "usage": {}, "attempts": 0}

    context = "\n\n".join(f"[Passage {idx + 1}]\n{chunk}" for idx, chunk in enumerate(chunks))
    prompt = f"""You are a question-answering assistant.

Answer the question using only the provided passages.
Return a concise English answer. If the passages do not contain enough information, answer "I don't know."
Do not include explanations, citations, or extra commentary.

Question:
{question}

Passages:
{context}

Answer:"""

    last = {"answer": "", "finish_reason": "", "usage": {}, "attempts": 0}
    for attempt in range(1, 4):
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions in concise English using only the provided evidence.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.1")),
            stream=False,
        )
        last = {
            "answer": (response.choices[0].message.content or "").strip(),
            "finish_reason": response.choices[0].finish_reason,
            "usage": response_usage(response),
            "attempts": attempt,
        }
        if last["answer"]:
            return last
        time.sleep(0.5)
    return last


def normalize_answer(text: str) -> str:
    """HotpotQA-style English normalization."""
    if not text:
        return ""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def calculate_item_metrics(prediction: str, ground_truth: str) -> tuple[float, float, float, int]:
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    em = int(pred_norm == gt_norm)

    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()
    if not pred_tokens or not gt_tokens:
        f1 = 1.0 if pred_tokens == gt_tokens else 0.0
        return 0.0, 0.0, f1, em

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0, em

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, em


def run_generation(label: str, input_path: Path, output_path: Path) -> list[dict]:
    client = make_client()
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    generated = []
    existing = {}
    if output_path.exists():
        try:
            existing = {item["id"]: item for item in json.loads(output_path.read_text(encoding="utf-8"))}
        except Exception:
            existing = {}

    for row in tqdm(rows, desc=f"generate {label}"):
        existing_item = existing.get(row["id"])
        if (
            existing_item
            and existing_item.get("generated_answer")
            and existing_item.get("finish_reason") in {"stop", "no_context"}
        ):
            generated.append(existing_item)
            continue

        chunks = row.get("retrieved_chunks", [])[:GENERATION_TOP_K]
        ids = row.get("ids", [])[:GENERATION_TOP_K]
        generation = generate_answer(client, row["question"], chunks)
        item = {
            "id": row["id"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "generated_answer": generation["answer"],
            "ground_truth_chunks": row.get("ground_truth_chunks", []),
            "ids": ids,
            "retrieved_chunks": chunks,
            "generation_model": GENERATION_MODEL,
            "finish_reason": generation["finish_reason"],
            "usage": generation["usage"],
            "attempts": generation["attempts"],
        }
        generated.append(item)
        output_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.2)

    output_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
    return generated


def evaluate(label: str, rows: list[dict], eval_path: Path) -> dict:
    eval_rows = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    em_scores = []

    for row in rows:
        precision, recall, f1, em = calculate_item_metrics(row.get("generated_answer", ""), row.get("gold_answer", ""))
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        em_scores.append(em)
        eval_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "generated_answer": row.get("generated_answer", ""),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "em": em,
            }
        )

    summary = {
        "method": label,
        "count": len(rows),
        "exact_match": float(np.mean(em_scores)),
        "f1": float(np.mean(f1_scores)),
        "precision": float(np.mean(precision_scores)),
        "recall": float(np.mean(recall_scores)),
    }
    payload = {"summary": summary, "items": eval_rows}
    eval_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summaries = {}
    for label, input_path in INPUT_FILES.items():
        generated = run_generation(label, input_path, OUTPUT_FILES[label])
        summaries[label] = evaluate(label, generated, EVAL_FILES[label])

    report = {}
    if REPORT_FILE.exists():
        try:
            report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
        except Exception:
            report = {}
    report["generation_model"] = GENERATION_MODEL
    report["generation_top_k"] = GENERATION_TOP_K
    report["generation_max_tokens"] = None
    report["generation_methods"] = summaries
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
