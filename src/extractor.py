"""Document text extraction module supporting PDFs, Markdown, and plain text.
Extracts page-by-page text with structural metadata and calculates document hash.
"""

import hashlib
import io
from pathlib import Path
from typing import List, Dict, Any, Union

def calculate_sha256(content: Union[str, bytes]) -> str:
    """Calculates SHA-256 hash of text or bytes for versioning and invalidation."""
    hasher = hashlib.sha256()
    if isinstance(content, str):
        hasher.update(content.encode("utf-8"))
    else:
        hasher.update(content)
    return hasher.hexdigest()

def extract_text_from_pdf(file_bytes_or_path: Union[bytes, str, Path], filename: str = "document.pdf") -> List[Dict[str, Any]]:
    """Extracts text page by page from a PDF file using PyMuPDF (fitz).
    
    Returns:
        List of dicts: [{"text": str, "page": int, "source": str, "doc_hash": str}]
    """
    try:
        import pymupdf as fitz  # PyMuPDF
    except ImportError:
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF extraction. Run `pip install pymupdf`.")

    if isinstance(file_bytes_or_path, (str, Path)):
        doc = fitz.open(str(file_bytes_or_path))
        raw_bytes = Path(file_bytes_or_path).read_bytes()
    else:
        doc = fitz.open(stream=file_bytes_or_path, filetype="pdf")
        raw_bytes = file_bytes_or_path

    doc_hash = calculate_sha256(raw_bytes)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:  # Only include non-empty pages
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": filename,
                "doc_hash": doc_hash
            })

    doc.close()
    return pages

def extract_text_from_txt(content: Union[str, bytes], filename: str = "document.txt") -> List[Dict[str, Any]]:
    """Extracts text from plain text or Markdown files."""
    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="replace").strip()
        doc_hash = calculate_sha256(content)
    else:
        text = content.strip()
        doc_hash = calculate_sha256(text)

    return [{
        "text": text,
        "page": 1,
        "source": filename,
        "doc_hash": doc_hash
    }]

def extract_documents(uploaded_file, filename: str) -> List[Dict[str, Any]]:
    """Universal extractor handling PDF, TXT, and Markdown."""
    if isinstance(uploaded_file, bytes):
        data = uploaded_file
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    elif isinstance(uploaded_file, Path):
        data = uploaded_file.read_bytes()
    elif isinstance(uploaded_file, str):
        # Check if it is a valid file path on disk
        p = Path(uploaded_file)
        try:
            if p.is_file():
                data = p.read_bytes()
            else:
                data = uploaded_file.encode("utf-8")
        except Exception:
            data = uploaded_file.encode("utf-8")
    else:
        data = str(uploaded_file).encode("utf-8")

    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf(data, filename=filename)
    else:
        return extract_text_from_txt(data, filename=filename)
