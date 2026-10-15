"""Shared knowledge-base parsing without model or API initialization."""

from pathlib import Path
import re


def read_kb_chunks(path: str | Path) -> tuple[list[str], list[str]]:
    """Read the existing [题_N_块_M] / --- format, preserving order and text."""
    ids, chunks = [], []
    for block in Path(path).read_text(encoding="utf-8").split("---"):
        block = block.strip()
        match = re.search(r"\[(题_\d+_块_\d+)\]", block)
        if not match:
            continue
        ids.append(match.group(1))
        chunks.append(re.sub(r"\[题_\d+_块_\d+\]", "", block).strip())
    return ids, chunks
