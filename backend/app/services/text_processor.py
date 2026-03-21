"""
Text processing service
=======================

Thin facade over :mod:`app.utils.file_parser` that provides the three
text operations needed by the knowledge-graph pipeline:

1. **Extraction** — read one or more PDF/Markdown/TXT files and return their
   combined plain-text content.
2. **Preprocessing** — normalise line endings, collapse excessive blank lines,
   and strip trailing whitespace so the text is ready for chunking.
3. **Chunking** — split a long string into overlapping fixed-size windows
   suitable for feeding to the Zep graph builder in batches.
"""

from typing import List, Optional
from ..utils.file_parser import FileParser, split_text_into_chunks


class TextProcessor:
    """Stateless collection of text-manipulation helpers.

    All methods are ``@staticmethod`` so they can be called without
    instantiation: ``TextProcessor.preprocess_text(raw)``.
    """
    
    @staticmethod
    def extract_from_files(file_paths: List[str]) -> str:
        """Extract and concatenate text from one or more files.

        Delegates to :meth:`~app.utils.file_parser.FileParser.extract_from_multiple`.
        Each file's content is labelled with a ``=== Document N: filename ===``
        header so the downstream LLM can identify provenance.

        Args:
            file_paths: Absolute or relative paths to PDF, Markdown, or TXT files.

        Returns:
            A single string containing all extracted text, separated by double
            newlines.
        """
        return FileParser.extract_from_multiple(file_paths)
    
    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Raw text
            chunk_size: Chunk size
            overlap: Overlap size
            
        Returns:
            List of text chunks
        """
        return split_text_into_chunks(text, chunk_size, overlap)
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text
        - Remove extra whitespace
        - Normalize line breaks
        
        Args:
            text: Raw text
            
        Returns:
            Processed text
        """
        import re
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove consecutive blank lines (keep at most two newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading and trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def get_text_stats(text: str) -> dict:
        """Return basic statistics about a text string.

        Returns:
            dict with keys ``total_chars``, ``total_lines``, and ``total_words``.
        """
        return {
            "total_chars": len(text),
            "total_lines": text.count('\n') + 1,
            "total_words": len(text.split()),
        }

