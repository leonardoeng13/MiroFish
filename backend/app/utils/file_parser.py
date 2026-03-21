"""
File parsing utilities
======================

Provides text extraction from the three document formats supported by
MiroFish: PDF, Markdown, and plain text.

Supported extensions: ``.pdf``, ``.md``, ``.markdown``, ``.txt``

PDF extraction
    Uses `PyMuPDF <https://pymupdf.readthedocs.io/>`_ (``import fitz``).
    Each page is read and joined with double newlines.

Markdown / TXT extraction
    Uses a multi-level encoding detection strategy to handle non-UTF-8
    files gracefully (charset_normalizer → chardet → UTF-8 with
    ``errors='replace'``).

Chunking
    :func:`split_text_into_chunks` implements a sentence-aware sliding window.
    It tries to break at sentence-ending punctuation (both CJK and Latin)
    before falling back to a hard character limit.  An ``overlap`` of 50
    characters ensures context is preserved across chunk boundaries.
"""

import os
from pathlib import Path
from typing import List, Optional


def _read_text_with_fallback(file_path: str) -> str:
    """
    Read a text file, falling back to automatic encoding detection if UTF-8 fails.
    
    Uses a multi-level fallback strategy:
    1. Try UTF-8 decoding first
    2. Use charset_normalizer to detect encoding
    3. Fall back to chardet for encoding detection
    4. Final fallback: UTF-8 with errors='replace'
    
    Args:
        file_path: Path to the file
        
    Returns:
        Decoded text content
    """
    data = Path(file_path).read_bytes()
    
    # Try UTF-8 first
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass
    
    # Try charset_normalizer for encoding detection
    encoding = None
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best and best.encoding:
            encoding = best.encoding
    except Exception:
        pass
    
    # Fall back to chardet
    if not encoding:
        try:
            import chardet
            result = chardet.detect(data)
            encoding = result.get('encoding') if result else None
        except Exception:
            pass
    
    # Final fallback: UTF-8 with replace
    if not encoding:
        encoding = 'utf-8'
    
    return data.decode(encoding, errors='replace')


class FileParser:
    """Extract plain text from supported document formats.

    All public methods are class-methods so the class can be used without
    instantiation.  Unsupported extensions raise :exc:`ValueError`; missing
    files raise :exc:`FileNotFoundError`.
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.markdown', '.txt'}
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """
        Extract text from a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        if suffix == '.pdf':
            return cls._extract_from_pdf(file_path)
        elif suffix in {'.md', '.markdown'}:
            return cls._extract_from_md(file_path)
        elif suffix == '.txt':
            return cls._extract_from_txt(file_path)
        
        raise ValueError(f"Cannot process file format: {suffix}")
    
    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required: pip install PyMuPDF")
        
        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def _extract_from_md(file_path: str) -> str:
        """Extract text from a Markdown file with automatic encoding detection"""
        return _read_text_with_fallback(file_path)
    
    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """Extract text from a TXT file with automatic encoding detection"""
        return _read_text_with_fallback(file_path)
    
    @classmethod
    def extract_from_multiple(cls, file_paths: List[str]) -> str:
        """
        Extract and merge text from multiple files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Merged text content
        """
        all_texts = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                text = cls.extract_text(file_path)
                filename = Path(file_path).name
                all_texts.append(f"=== Document {i}: {filename} ===\n{text}")
            except Exception as e:
                all_texts.append(f"=== Document {i}: {file_path} (extraction failed: {str(e)}) ===")
        
        return "\n\n".join(all_texts)


def split_text_into_chunks(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50
) -> List[str]:
    """
    Split text into smaller chunks
    
    Args:
        text: Source text
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to split at a sentence boundary
        if end < len(text):
            # Find the nearest sentence-ending punctuation
            for sep in ['。', '！', '？', '.\n', '!\n', '?\n', '\n\n', '. ', '! ', '? ']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Next chunk starts from the overlap position
        start = end - overlap if end < len(text) else len(text)
    
    return chunks
