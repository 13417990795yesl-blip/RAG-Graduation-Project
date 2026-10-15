"""Repository-relative paths; load private local configuration without overrides."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / "config" / ".env"
load_dotenv(ENV_FILE, override=False)


def configured_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


BASE_MODEL_DIR = configured_path(
    "BGE_MODEL_PATH",
    PROJECT_ROOT / "models/BAAI/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
)
RERANKER_DIR = configured_path(
    "BGE_RERANKER_PATH",
    PROJECT_ROOT / "models/BAAI/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
)
