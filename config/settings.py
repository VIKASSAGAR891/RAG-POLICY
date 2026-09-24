from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def _int(name, default):
    return int(os.getenv(name, default))

def _float(name, default):
    return float(os.getenv(name, default))

@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    raw_data_dir: Path = ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
    processed_data_dir: Path = ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
    index_dir: Path = ROOT / os.getenv("INDEX_DIR", "storage/index")
    eval_dir: Path = ROOT / os.getenv("EVAL_DIR", "data/evaluation")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.0)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
    dense_top_k: int = _int("DENSE_TOP_K", 20)
    bm25_top_k: int = _int("BM25_TOP_K", 20)
    rerank_top_k: int = _int("RERANK_TOP_K", 6)
    rrf_k: int = _int("RRF_K", 60)
    chunk_max_chars: int = _int("CHUNK_MAX_CHARS", 6000)
    chunk_min_chars: int = _int("CHUNK_MIN_CHARS", 300)

    @property
    def groq_api_key(self):
        return os.getenv("GROQ_API_KEY", "")

SETTINGS = Settings()
