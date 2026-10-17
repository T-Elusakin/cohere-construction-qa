from fastapi.testclient import TestClient

import api
from src.loader import Chunk

client = TestClient(api.app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(mocker):
    fake_chunk = Chunk(
        text="Hard hats must be worn on site.", source="hsg150.pdf", chunk_index=0
    )
    mocker.patch.object(api, "get_store", return_value="fake store")
    mocker.patch.object(api, "retrieve", return_value=[(fake_chunk, 0.9)])
    mocker.patch.object(
        api,
        "generate_answer",
        return_value=("Hard hats are required.", ["hsg150.pdf"]),
    )

    response = client.post("/ask", json={"question": "What PPE is required?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Hard hats are required.",
        "sources": ["hsg150.pdf"],
    }


def test_ask_rejects_empty_question(mocker):
    mock_retrieve = mocker.patch.object(api, "retrieve")

    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 400
    assert response.json() == {"detail": "Question must not be empty."}
    mock_retrieve.assert_not_called()
