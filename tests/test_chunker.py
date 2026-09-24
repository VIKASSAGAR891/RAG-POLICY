from pathlib import Path
from src.policy_qa.chunker import build_chunks

def test_structural_chunking_keeps_sections():
    pages = [{
        "page": 1,
        "text": "1 GENERAL TERMS\nThis is the main rule.\n\n1.1 Exception\nThis is an exception."
    }]
    chunks = build_chunks(Path("demo.pdf"), pages, max_chars=1000, min_chars=1)
    assert len(chunks) >= 2
    assert any(c.section_id == "1.1" for c in chunks)
