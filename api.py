from pathlib import Path 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.generate import generate_answer
from src.loader import load_and_chunk_pdfs
from src.retrieval import retrieve
from src.store import VectorStore   

PDF_DIR = Path("data/pdfs")

app = FastAPI(title="Construction Document Q&A")

store = None 

def get_store() -> VectorStore: 
    global store 
    if store is None: 
        chunks = load_and_chunk_pdfs(PDF_DIR)
        store = VectorStore.build(chunks)
    return store

class AskRequest(BaseModel): 
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest): 
    if not request.question.strip(): 
        raise HTTPException(status_code=400, detail="Question must not be empty.")    
    
    results = retrieve(request.question, get_store())
    top_chunks = [chunk for chunk, _ in results]
    answer, sources = generate_answer(request.question, top_chunks)
    return AskResponse(answer=answer, sources=sources)