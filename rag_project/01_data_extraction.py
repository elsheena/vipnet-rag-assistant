"""
01_data_extraction.py
---------------------
Extracts text from ViPNet Coordinator HW 5 PDF documentation,
applies cleaning and chunking, and saves the result to JSON.
"""

import os
import json
import re
from pathlib import Path

import fitz  # PyMuPDF


DOCS_DIR = Path(__file__).parent.parent / "ViPNet Coordinator HW 5.3.2_docs"
OUTPUT_FILE = Path(__file__).parent / "data" / "chunks.json"
OUTPUT_FILE.parent.mkdir(exist_ok=True)

# Chunking parameters
CHUNK_SIZE = 1000       # characters
CHUNK_OVERLAP = 200     # characters


def clean_text(text: str) -> str:
    """Remove excessive whitespace, headers, footers, and fix hyphenation."""
    # Remove header/footer lines like "ViPNet Coordinator HW 5... | 241"
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip lines that look like page headers/footers (contain "|" and digits)
        if "|" in line and any(c.isdigit() for c in line):
            continue
        # Skip lines that are just page numbers
        if line.strip().isdigit():
            continue
        cleaned_lines.append(line)
    
    text = " ".join(cleaned_lines)

    # Remove soft hyphens and zero-width spaces
    text = text.replace("\xad", "").replace("\u200b", "")
    # Fix hyphenated words at end of line (e.g. "конфи-\nгурация" -> "конфигурация")
    text = re.sub(r"-\s+", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF, returning list of page dicts."""
    pages = []
    doc = fitz.open(str(pdf_path))
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = clean_text(text)
        if text:
            pages.append({
                "source": pdf_path.name,
                "page": page_num,
                "text": text,
            })
    doc.close()
    return pages


def split_into_chunks(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split page texts into overlapping character-level chunks.
    Each chunk retains source metadata.
    """
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks),
                    "source": page["source"],
                    "page": page["page"],
                    "text": chunk_text,
                })
            if end == len(text):
                break
            start += chunk_size - overlap
    return chunks


def main():
    all_chunks = []
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {DOCS_DIR}")

    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        pages = extract_pages(pdf_path)
        chunks = split_into_chunks(pages)
        all_chunks.extend(chunks)
        print(f"    -> {len(pages)} pages, {len(chunks)} chunks")

    # Re-index chunk IDs globally
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
