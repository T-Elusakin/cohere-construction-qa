from unittest.mock import MagicMock

from src import generate
from src.loader import Chunk


def test_generate_answer_includes_source_citation(mocker):
    chunks = [
        Chunk(text="Hard hats must be worn on site.", source="hsg150.pdf", chunk_index=0),
    ]

    fake_content_item = MagicMock()
    fake_content_item.text = "Hard hats are required on site."

    fake_source = MagicMock()
    fake_source.id = "0"  # matches the doc id generate_answer assigns chunks[0]

    fake_citation = MagicMock()
    fake_citation.sources = [fake_source]

    fake_response = MagicMock()
    fake_response.message.content = [fake_content_item]
    fake_response.message.citations = [fake_citation]

    mocker.patch.object(generate.co, "chat", return_value=fake_response)

    answer, sources = generate.generate_answer("What PPE is required?", chunks)

    assert answer == "Hard hats are required on site."
    assert sources == ["hsg150.pdf"]
