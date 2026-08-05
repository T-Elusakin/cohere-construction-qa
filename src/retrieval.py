import os

import cohere
from dotenv import load_dotenv

from src.loader import Chunk 
from src.store import VectorStore, embed_texts 

load_dotenv()
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

RERANK_MODEL = "rerank-english-v3.0"
INITIAL_CANDIDATES = 20
FINAL_RESULTS = 5

def retrieve(query: str, store: VectorStore) -> list[tuple[Chunk,float]]: # takes a query string and a VectorStore object, and returns a list of tuples containing the top chunks and their corresponding similarity scores. The function first embeds the query using the embed_texts function, then searches the VectorStore for the top candidates based on the query embedding. Finally, it reranks the candidates using the Cohere rerank model and returns the top results.
    query_embedding = embed_texts([query], input_type = "search_query")[0] # embeds the query string into a vector representation using the embed_texts function, specifying that the input type is a search query. The result is a single embedding vector for the query.
    candidates = store.search(query_embedding, top_k=INITIAL_CANDIDATES) # searches the VectorStore for the top INITIAL_CANDIDATES chunks that are most similar to the query embedding. The search method returns a list of tuples, each containing a chunk and its corresponding similarity score.

    documents = [chunk.text for chunk, _ in candidates]
    response = co.rerank(
        model=RERANK_MODEL, # specifies the reranking model to use
        query=query, # the original query string to rerank against
        documents=documents, # the list of candidate document texts to be reranked
        top_n=FINAL_RESULTS, # specifies how many top results to return after reranking
    )

    results = []
    for item in response.results: 
        chunk, _ = candidates[item.index]
        results.append((chunk, item.relevance_score))
    return results 




