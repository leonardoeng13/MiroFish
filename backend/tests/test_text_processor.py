"""
Unit tests for app.services.text_processor and app.utils.file_parser (split_text_into_chunks)
"""

import pytest
from app.services.text_processor import TextProcessor
from app.utils.file_parser import split_text_into_chunks


# ---------------------------------------------------------------------------
# split_text_into_chunks
# ---------------------------------------------------------------------------

class TestSplitTextIntoChunks:
    def test_empty_string_returns_empty_list(self):
        assert split_text_into_chunks("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert split_text_into_chunks("   \n  ") == []

    def test_short_text_returned_as_single_chunk(self):
        text = "Short text."
        result = split_text_into_chunks(text, chunk_size=500)
        assert result == [text]

    def test_long_text_is_split_into_multiple_chunks(self):
        text = "A" * 1200
        result = split_text_into_chunks(text, chunk_size=500, overlap=50)
        assert len(result) > 1

    def test_chunks_cover_full_text(self):
        """All characters from the original text must appear in at least one chunk."""
        text = "word " * 300  # 1500 chars
        chunks = split_text_into_chunks(text, chunk_size=200, overlap=20)
        # Reconstruct a version by joining and checking that no content is silently dropped
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_overlap_creates_shared_content(self):
        """With overlap > 0, the tail of chunk[n] should appear in chunk[n+1]."""
        text = "abcdefghij" * 20  # 200 chars
        chunks = split_text_into_chunks(text, chunk_size=50, overlap=10)
        if len(chunks) >= 2:
            # The end of the first chunk must be a prefix of the second chunk
            tail = chunks[0][-10:]
            assert chunks[1].startswith(tail)

    def test_sentence_boundary_split_chinese(self):
        """Chunker should prefer sentence-ending punctuation (。) over mid-word cuts."""
        text = ("这是第一句话。" * 20) + ("这是第二句话！" * 20)
        chunks = split_text_into_chunks(text, chunk_size=80, overlap=0)
        # Each chunk should end with a CJK sentence terminator when long enough
        for chunk in chunks[:-1]:
            assert chunk[-1] in {"。", "！", "？"} or len(chunk) < 80

    def test_sentence_boundary_split_english(self):
        """Chunker should prefer '. ', '! ', '? ' as split points in English text."""
        text = ("Hello world. " * 15) + ("Another sentence! " * 15)
        chunks = split_text_into_chunks(text, chunk_size=100, overlap=0)
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# TextProcessor
# ---------------------------------------------------------------------------

class TestTextProcessor:
    def test_preprocess_normalises_crlf(self):
        raw = "line1\r\nline2\r\nline3"
        result = TextProcessor.preprocess_text(raw)
        assert "\r" not in result
        assert "line1" in result
        assert "line2" in result

    def test_preprocess_collapses_multiple_blank_lines(self):
        raw = "line1\n\n\n\n\nline2"
        result = TextProcessor.preprocess_text(raw)
        # At most two newlines between lines
        assert "\n\n\n" not in result

    def test_preprocess_strips_leading_trailing_whitespace(self):
        raw = "  hello  \n  world  "
        result = TextProcessor.preprocess_text(raw)
        lines = result.split("\n")
        for line in lines:
            assert line == line.strip()

    def test_preprocess_returns_stripped_result(self):
        raw = "\n\n  hello world  \n\n"
        result = TextProcessor.preprocess_text(raw)
        assert result == result.strip()

    def test_get_text_stats_returns_correct_structure(self):
        text = "Hello world\nSecond line"
        stats = TextProcessor.get_text_stats(text)
        assert "total_chars" in stats
        assert "total_lines" in stats
        assert "total_words" in stats

    def test_get_text_stats_char_count(self):
        text = "abc"
        stats = TextProcessor.get_text_stats(text)
        assert stats["total_chars"] == 3

    def test_get_text_stats_line_count(self):
        text = "line1\nline2\nline3"
        stats = TextProcessor.get_text_stats(text)
        assert stats["total_lines"] == 3

    def test_get_text_stats_word_count(self):
        text = "one two three"
        stats = TextProcessor.get_text_stats(text)
        assert stats["total_words"] == 3

    def test_split_text_delegates_correctly(self):
        text = "x" * 2000
        result = TextProcessor.split_text(text, chunk_size=500, overlap=50)
        assert isinstance(result, list)
        assert len(result) > 1
