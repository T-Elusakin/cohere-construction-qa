# Main code - doesnt contain any new logic of its own, but it orchestrates the enture pipiline we've written
from pathlib import Path 

from src.generate import generate_answer
from src.loader import load_and_chunk_pdfs
from src.retrieval import retrieve
from src.store import VectorStore

PDF_DIR = Path("data/pdfs")

def main() -> None: 
    print("Loading documents...")
    chunks = load_and_chunk_pdfs(PDF_DIR)
    store = VectorStore.build(chunks)
    print(f"Ready - {len(chunks)} chunks loaded from {PDF_DIR}.")
    print("Ask a question about the construction/safety documents (or type 'exit' to quit).\n")
    # loading/chunking/embedding work exactly once, before the loop starts - not once per question. Thanks to the disk cache in store.py

    # The interactive loop: 
    while True: 
        query = input ("> ").strip()
        if not query: # if the user presses enter without typing anything or only typiing white space, query remains True so it skips the rest of the loop and asks again 
            continue 
        if query.lower() in {"exit", "quit"}: 
            break

        results = retrieve(query, store)
        top_chunks = [chunk for chunk, _ in results]
        answer, sources = generate_answer(query, top_chunks)

        print(f"\n{answer}\n")
        if sources:
            print(f"Sources: {', '.join(sources)}\n")

if __name__ == "__main__": # only actually start the interactive loop when someone runs this file directly (python main.py)
            main() 