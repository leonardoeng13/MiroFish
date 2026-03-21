"""
Unit tests for app.utils.file_parser.FileParser
"""

import os
import pytest
from app.utils.file_parser import FileParser


class TestFileParser:
    # ------------------------------------------------------------------
    # extract_text – error paths
    # ------------------------------------------------------------------

    def test_raises_for_nonexistent_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "ghost.txt")
        with pytest.raises(FileNotFoundError):
            FileParser.extract_text(path)

    def test_raises_for_unsupported_extension(self, tmp_dir):
        path = os.path.join(tmp_dir, "file.docx")
        with open(path, "w") as f:
            f.write("content")
        with pytest.raises(ValueError, match="Unsupported file format"):
            FileParser.extract_text(path)

    # ------------------------------------------------------------------
    # extract_text – TXT
    # ------------------------------------------------------------------

    def test_extracts_utf8_txt(self, sample_txt_file):
        path, expected = sample_txt_file
        result = FileParser.extract_text(path)
        assert "Hello, world!" in result
        assert "test file" in result

    def test_extracts_latin1_txt(self, tmp_dir):
        path = os.path.join(tmp_dir, "latin1.txt")
        content = "café résumé"
        with open(path, "w", encoding="latin-1") as f:
            f.write(content)
        result = FileParser.extract_text(path)
        assert result is not None
        assert len(result) > 0

    # ------------------------------------------------------------------
    # extract_text – Markdown
    # ------------------------------------------------------------------

    def test_extracts_md_file(self, sample_md_file):
        path, expected = sample_md_file
        result = FileParser.extract_text(path)
        assert "Title" in result
        assert "Paragraph" in result

    def test_extracts_markdown_extension(self, tmp_dir):
        path = os.path.join(tmp_dir, "doc.markdown")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Header\n\nBody text.")
        result = FileParser.extract_text(path)
        assert "Header" in result

    # ------------------------------------------------------------------
    # extract_from_multiple
    # ------------------------------------------------------------------

    def test_extract_from_multiple_combines_files(self, tmp_dir):
        paths = []
        for i in range(3):
            p = os.path.join(tmp_dir, f"doc{i}.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(f"Content of document {i}.")
            paths.append(p)
        result = FileParser.extract_from_multiple(paths)
        for i in range(3):
            assert f"Content of document {i}" in result

    def test_extract_from_multiple_handles_bad_file_gracefully(self, tmp_dir):
        good = os.path.join(tmp_dir, "good.txt")
        with open(good, "w", encoding="utf-8") as f:
            f.write("Good content.")
        bad = os.path.join(tmp_dir, "missing.txt")  # does not exist
        result = FileParser.extract_from_multiple([good, bad])
        assert "Good content" in result
        assert "extraction failed" in result  # graceful failure message

    # ------------------------------------------------------------------
    # SUPPORTED_EXTENSIONS
    # ------------------------------------------------------------------

    def test_supported_extensions_contains_expected(self):
        assert ".pdf" in FileParser.SUPPORTED_EXTENSIONS
        assert ".md" in FileParser.SUPPORTED_EXTENSIONS
        assert ".txt" in FileParser.SUPPORTED_EXTENSIONS
        assert ".markdown" in FileParser.SUPPORTED_EXTENSIONS
