# Imports and a data structure to hold each chunk of text from the pdfs

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

@dataclass
class Chunk: 
    text: str
    source: str
    chunk_index: int 

# Extracting raw text from a pdf
def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)

# Splitting the text into overlapping chunks 
def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    words = text.split()
    chunks = []
    start = 0 
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words): 
            break
        start = end - overlap
    return chunks

# Tying it together for a whole folder of PDFs
def load_and_chunk_pdfs(pdf_dir: Path, chunk_size: int = 150, overlap: int = 30) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        text = extract_text_from_pdf(pdf_path)
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            all_chunks.append(Chunk(text=chunk, source=pdf_path.name, chunk_index=i))
    return all_chunks
    