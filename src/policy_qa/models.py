from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class Chunk:
    chunk_id: str
    text: str
    document: str
    page_start: int
    page_end: int
    section_id: str = ""
    section_title: str = ""
    parent_section: str = ""

    def to_dict(self):
        return asdict(self)

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    method: str = ""
    rerank_score: Optional[float] = None
