import re
from pathlib import Path
from .models import Chunk

HEADING_PATTERNS = [
    re.compile(r"^\s*(ARTICLE\s+[IVXLC]+(?:\.?|\s.*))\s*$", re.I),
    re.compile(r"^\s*(SECTION\s+\d+(?:\.\d+)*(?:\s*[-–—:]?\s*.*)?)\s*$", re.I),
    re.compile(r"^\s*(\d+(?:\.\d+){0,5})\s+[A-Z][^\n]{1,180}$"),
    re.compile(r"^\s*([A-Z][A-Z0-9][A-Z0-9\s,&/()'’:-]{3,120})\s*$"),
]

def is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 220:
        return False
    return any(p.match(line) for p in HEADING_PATTERNS)

def normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_long(text: str, max_chars: int):
    if len(text) <= max_chars:
        return [text]
    paras = re.split(r"\n\s*\n", text)
    pieces, current = [], ""
    for para in paras:
        if not para.strip():
            continue
        candidate = (current + "\n\n" + para).strip() if current else para.strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            if len(para) <= max_chars:
                current = para.strip()
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para.strip())
                current = ""
                buf = ""
                for s in sentences:
                    if len(buf) + len(s) + 1 <= max_chars:
                        buf = (buf + " " + s).strip()
                    else:
                        if buf:
                            pieces.append(buf)
                        buf = s
                current = buf
    if current:
        pieces.append(current)
    return pieces

def build_chunks(pdf_path: Path, pages, max_chars=6000, min_chars=300):
    chunks = []
    current_lines = []
    section_stack = []
    chunk_start_page = 1
    counter = 0

    def flush(end_page):
        nonlocal counter, current_lines, chunk_start_page
        text = normalize("\n".join(current_lines))
        if not text:
            current_lines = []
            return
        for piece_idx, piece in enumerate(split_long(text, max_chars)):
            if len(piece) < min_chars and chunks:
                # Keep small trailing fragments attached to their section rather than
                # creating noisy micro-chunks.
                chunks[-1].text += "\n\n" + piece
                chunks[-1].page_end = end_page
                continue
            counter += 1
            section_id = section_stack[-1][0] if section_stack else ""
            section_title = section_stack[-1][1] if section_stack else ""
            parent = section_stack[-2][1] if len(section_stack) > 1 else ""
            chunks.append(Chunk(
                chunk_id=f"{pdf_path.stem}_{counter:05d}",
                text=piece,
                document=pdf_path.name,
                page_start=chunk_start_page,
                page_end=end_page,
                section_id=section_id,
                section_title=section_title,
                parent_section=parent,
            ))
        current_lines = []

    for page in pages:
        page_no = page["page"]
        lines = page["text"].splitlines()
        if not current_lines:
            chunk_start_page = page_no
        for raw in lines:
            line = raw.strip()
            if not line:
                current_lines.append("")
                continue
            if is_heading(line):
                # Flush before a new section so rules and exceptions stay attached
                # to the section that introduced them.
                flush(page_no)
                m = re.match(r"^\s*(\d+(?:\.\d+){0,5})\s+(.*)$", line)
                if m:
                    sec_id, title = m.group(1), m.group(2).strip()
                    depth = sec_id.count(".") + 1
                else:
                    sec_id, title = line[:80], line
                    depth = 1
                section_stack[:] = section_stack[:max(0, depth-1)]
                section_stack.append((sec_id, title))
                current_lines.append(line)
            else:
                current_lines.append(line)
    flush(pages[-1]["page"] if pages else chunk_start_page)
    return chunks
