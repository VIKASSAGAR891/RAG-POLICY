from pathlib import Path
import fitz

def extract_pdf_pages(pdf_path: Path):
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append({
                "page": page_no,
                "text": text or "",
            })
    return pages
