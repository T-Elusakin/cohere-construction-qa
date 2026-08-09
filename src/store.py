# Imports, client setup, and constants 
import hashlib # for generating unique identifiers for each chunk
import os
from pathlib import Path

import cohere
import numpy as np
from dotenv import load_dotenv

from src.loader import Chunk

load_dotenv()
# Initialize the Cohere client with the API key from environment variables
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

EMBED_MODEL = "embed-english-v3.0"  # Model for embedding
BATCH_SIZE = 96
CACHE_DIR = Path("data/cache")  # Directory for caching embeddings

# embedding in batches 
def embed_texts(texts: list[str], input_type: str) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE): 
        batch = texts [i:i + BATCH_SIZE]
        response = co.embed(
            model=EMBED_MODEL, # picks which embedding model to use
            input_type=input_type, # tells us what kind of input we're embedding (search_document, query, etc.)
            texts=batch, # which texts to embed
            embedding_types=["float"],# which format the embeddings should be returned in (float, int8, etc.
        )
        all_embeddings.extend(response.embeddings.float) # adds each embedding from the response into the list
    return np.array(all_embeddings, dtype=np.float32) # converts the collected list into a NumPy array, using 32-bit floating point numbers 

# Helper functions: normalising vectors, generating unique identifiers, and caching embeddings
def _normalise(vector: np.array) -> np.array: # This function normalizes a vector to have a unit length (magnitude of 1). This is often done in machine learning and information retrieval to ensure that the scale of the vectors does not affect similarity calculations. The normalization is done by dividing each element of the vector by its Euclidean norm (length).
    norms = np.linalg.norm(vector, axis=1, keepdims=True) # computes the length or magnitude of each row in the vector array. The axis=1 means "do it row by row", and keepdims=True keeps the output in a 2D shape, which can be divided nicely.
    return vector / norms # divided the vector by the length, this is called normalisation (remember linear algebra vector normalisation is dividing a vector by its length to make it a unit vector). This is useful for comparing vectors in a consistent way, especially in tasks like similarity search.

def chunks_hash(chunks: list[Chunk]) -> str: # takes a list of chunk objects and returns a string that is a unique identifier for the list of chunks. This is useful for caching, as it allows us to check if we've already processed this exact set of chunks before.
    combined = "".join(f"{c.source}:{c.chunk_index}:{c.text}" for c in chunks) # builds one long long straing by combining the chunk source, the chunk index and the chunk text for each chunk in the list. This ensures that even if two chunks have the same text but come from different sources or have different indices, they will produce a different combined string.
    return hashlib.sha256(combined.encode("utf-8")).hexdigest() # converts the string text into bytes so it can be hashed safely 

# The VectorStore class and cache-aware build()
# The VectorStore class is designed to store chunks of text along with their corresponding embeddings. It provides a method to build the store from a list of chunks, caching the embeddings for efficiency. If the same chunks are processed again, it can load the cached embeddings instead of recomputing them, saving time and resources.
class VectorStore: 
    def __init__(self, chunks: list[Chunk], embeddings:np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings 

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "VectorStore": # takes a list of chunks and returns a VectorStore object 
        CACHE_DIR.mkdir(parents=True, exist_ok=True) # creates the cache directory if it doesn't already exist
        hash_file = CACHE_DIR/"hash.txt" # creates a file path for storing the hash
        embeddings_file = CACHE_DIR/"embeddings.npy" # creates a file path for storing the embeddings on disk 
        current_hash = chunks_hash(chunks) # generates a unique identifier for the list of chunks

        if hash_file.exists() and embeddings_file.exists(): # checks if the cache files exist
            if hash_file.read_text().strip() == current_hash: # checks if the hash of the current chunks matches the hash stored in the cache
                print("Loading cached embeddings...")
                return cls(chunks, np.load(embeddings_file)) # loads the cached embeddings from the file and returns a new instance of VectorStore with the chunks and embeddings

        print(f"Embedding {len(chunks)} chunks via Cohere...") # if no valid cache is found, it prints that embeddings are being created for the chunks
        embeddings = _normalise(embed_texts([c.text for c in chunks], "search_document")) # generates embeddings for the chunks and normalizes them

        np.save(embeddings_file, embeddings) # saves the embeddings to a file for future use
        hash_file.write_text(current_hash) 
        return cls(chunks, embeddings) # returns a new instance of VectorStore with the chunks and embeddings

    # The search method takes a query string and returns the top k most similar chunks based on cosine similarity. It first generates an embedding for the query, normalizes it, and then computes the dot product with the stored embeddings to find the most similar chunks. The results are sorted by similarity score and returned as a list of tuples containing the chunk and its corresponding score.
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[Chunk, float]]: # defining a method called search where it takes an already created embedding for the query and top_k is the number of results you want back. The method returns a list of tuples, where each tuple contains a chunk and its corresponding similarity score.
        query_embedding = query_embedding / np.linalg.norm(query_embedding) # normalizes the query embedding to have a unit length
        scores = self.embeddings @ query_embedding # This compares the stored chunk embeddings with the query embedding. The @ operator performs a dot-product style comparison, which is a common way to measure similarity between vectors. The result is an array of scores, where each score corresponds to how similar a chunk is to the query.
        top_indices = np.argsort(scores)[::-1][:top_k] # This finds the best matching chunk positions by sorting the scores in ascending order and then [::-1 reverses the order so the highest scores come first and [::top_k] keeps only the top few results.]
        return [(self.chunks[i], float(scores[i])) for i in top_indices] # This returns the actual chunks that matched best and converts each score to a plain Python float for easier handling. The result is a list of tuples, where each tuple contains a chunk and its corresponding similarity score.

        
    