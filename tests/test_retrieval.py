from unittest.mock import MagicMock

import numpy as np

from src import retrieval
from src.loader import Chunk
from src.store import VectorStore


def _make_store() -> VectorStore:
    chunks = [
        Chunk(text="Hard hats must be worn on site.", source="a.pdf", chunk_index=0),
        Chunk(text="The weather was sunny today.", source="a.pdf", chunk_index=1),
        Chunk(text="Steel-toe boots are required.", source="a.pdf", chunk_index=2),
    ]
    # orthogonal unit vectors, so a query pointing at [1, 0, 0] matches
    # chunk 0 exactly and the others not at all - keeps vector search
    # deterministic without needing real embeddings.
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    return VectorStore(chunks, embeddings)


def test_retrieve_returns_expected_number_of_results(mocker):
    store = _make_store()
    mocker.patch.object(
        retrieval, "embed_texts", return_value=np.array([[1.0, 0.0, 0.0]])
    )

    fake_response = MagicMock()
    fake_response.results = [
        MagicMock(index=0, relevance_score=0.9),
        MagicMock(index=2, relevance_score=0.4),
    ]
    mocker.patch.object(retrieval.co, "rerank", return_value=fake_response)

    results = retrieval.retrieve("what PPE is required?", store)

    assert len(results) == 2
    assert results[0][0].source == "a.pdf"
    assert results[0][1] == 0.9


def test_retrieve_handles_zero_results_without_crashing(mocker):
    store = _make_store()
    mocker.patch.object(
        retrieval, "embed_texts", return_value=np.array([[1.0, 0.0, 0.0]])
    )

    fake_response = MagicMock()
    fake_response.results = []
    mocker.patch.object(retrieval.co, "rerank", return_value=fake_response)

    results = retrieval.retrieve("irrelevant query", store)

    assert results == []
