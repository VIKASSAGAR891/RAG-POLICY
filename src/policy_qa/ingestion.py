import json
from pathlib import Path

from tqdm import tqdm

from config.settings import SETTINGS
from .pdf_parser import extract_pdf_pages
from .chunker import build_chunks
from .embeddings import BGEEmbedder
from .index import HybridIndex


def discover_pdfs(raw_dir: Path):
    """
    Discover PDFs recursively.

    For the CUAD dataset, PDFs are stored under:
        data/raw/full_contract_pdf/

    If that directory exists, use it explicitly.
    Otherwise, fall back to recursively searching raw_dir.
    """
    pdf_root = raw_dir / "full_contract_pdf"

    if pdf_root.is_dir():
        return sorted(pdf_root.rglob("*.pdf"))

    return sorted(raw_dir.rglob("*.pdf"))


def ingest():
    """
    Ingest all PDF documents:
        PDFs
        -> page extraction
        -> structural chunking
        -> embeddings
        -> FAISS + BM25 hybrid index
    """

    SETTINGS.raw_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdfs = discover_pdfs(SETTINGS.raw_data_dir)

    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found under {SETTINGS.raw_data_dir}. "
            "Add your documents and rerun."
        )

    print(f"Found {len(pdfs)} PDF files.")

    all_chunks = []
    manifest = []

    for pdf in tqdm(pdfs, desc="Parsing PDFs"):
        pages = extract_pdf_pages(pdf)

        chunks = build_chunks(
            pdf,
            pages,
            max_chars=SETTINGS.chunk_max_chars,
            min_chars=SETTINGS.chunk_min_chars,
        )

        all_chunks.extend(chunks)

        manifest.append(
            {
                "document": pdf.name,
                "path": str(
                    pdf.relative_to(SETTINGS.raw_data_dir)
                ),
                "pages": len(pages),
                "chunks": len(chunks),
            }
        )

    print(
        f"Created {len(all_chunks)} chunks "
        f"from {len(pdfs)} PDFs."
    )

    SETTINGS.processed_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SETTINGS.processed_data_dir / "manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Loading embedding model...")

    embedder = BGEEmbedder(
        SETTINGS.embedding_model
    )

    print("Building FAISS + BM25 hybrid index...")

    index = HybridIndex(
        SETTINGS.index_dir,
        embedder,
        SETTINGS.rrf_k,
    )

    count = index.build(all_chunks)

    print(
        f"Indexed {len(pdfs)} PDFs into {count} chunks."
    )


if __name__ == "__main__":
    ingest()