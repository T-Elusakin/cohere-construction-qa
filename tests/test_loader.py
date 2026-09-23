from src.loader import chunk_text


def _make_text(word_count: int) -> str:
    return " ".join(f"word{i}" for i in range(word_count))


def test_chunks_land_near_target_size():
    text = _make_text(500)

    chunks = chunk_text(text, chunk_size=150, overlap=30)

    assert len(chunks) > 1
    # every chunk except possibly the last should be exactly chunk_size words
    for chunk in chunks[:-1]:
        assert len(chunk.split()) == 150
    # the last chunk just holds whatever words are left over, so it's <= chunk_size
    assert len(chunks[-1].split()) <= 150


def test_consecutive_chunks_overlap():
    text = _make_text(500)

    chunks = chunk_text(text, chunk_size=150, overlap=30)

    first_words = chunks[0].split()
    second_words = chunks[1].split()
    # the last 30 words of chunk 0 should reappear as the first 30 words of chunk 1
    assert first_words[-30:] == second_words[:30]


def test_empty_input_returns_no_chunks():
    assert chunk_text("", chunk_size=150, overlap=30) == []
