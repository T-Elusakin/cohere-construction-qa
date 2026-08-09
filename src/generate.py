# setup 
import os

import cohere 
from dotenv import load_dotenv

from src.loader import Chunk 

load_dotenv()
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

CHAT_MODEL = "command-a-03-2025"

# building the documents Cohere will ground its answer in

def generate_answer(query:str, chunks: list[Chunk]) -> tuple[str, list[str]]:
    id_to_chunk = {str(i): chunk for i, chunk in enumerate(chunks)}
    documents = [
        {"id": doc_id, "data": {"text": chunk.text, "source": chunk.source}}
        for doc_id, chunk in id_to_chunk.items()
        ]

    response = co.chat(
        model=CHAT_MODEL, 
        messages=[{"role": "user", "content": query}],
        documents=documents,
    )

    answer = response.message.content[0].text
    cited_sources = set()
    for citation in response.message.citations or []:
        for source in citation.sources or []: 
            chunk = id_to_chunk.get(source.id)
            if chunk: 
                cited_sources.add(chunk.source)

    return answer, sorted(cited_sources)
