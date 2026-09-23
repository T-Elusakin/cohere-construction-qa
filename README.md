# Construction Site Report Q&A

A command-line tool that answers natural-language questions about construction
safety documents, with every answer grounded in — and cited back to — the
actual source text. Built end-to-end on Cohere's API: `embed` for semantic
search, `rerank` for precision, and `chat` (Command A) for grounded generation.

```
> what PPE is required on site?

The following PPE should be worn on site:
- Hard hats
- Wellington boots
- Overalls
- Gloves
- Ear defenders
- Goggles
- Harnesses
- Respirators

Sources: hsg150.pdf
```

## Why I built this

I have a First-Class Civil Engineering degree and wrote my dissertation on
generative AI in complex construction decision-making, where I found that one of the biggest barriers to generative AI adoption in the construction industry is trust. I'm starting an MSc in AI
Applications and Innovation in September 2026, and this project is a
deliberately small, end-to-end demonstration of retrieval-augmented generation
(RAG) applied to a domain I know well: on real UK Health and Safety Executive
(HSE) construction guidance, where "the model made it up" isn't an acceptable
answer — every claim needs to be traceable back to an actual source document.

## How it works

This is a **retrieve-then-rerank** RAG pipeline, not just "stuff a document
into a prompt":

1. **Load & chunk** (`src/loader.py`) — PDFs are parsed with `pypdf` and split
   into overlapping ~150-word chunks, so a fact sitting at a chunk boundary
   isn't accidentally cut in half.
2. **Embed & store** (`src/store.py`) — every chunk is embedded with Cohere's
   `embed-english-v3.0` model into a 1024-dimension vector. Vectors are kept
   in memory as a single normalized NumPy matrix (no external vector database
   needed at this scale), and cached to disk — a hash of the chunk contents
   determines when the cache is stale and needs recomputing, so repeated runs
   don't re-call the API unnecessarily.
3. **Retrieve** (`src/retrieval.py`) — a question is embedded the same way,
   compared against all stored chunks via cosine similarity to cheaply find
   the top 20 candidates, then Cohere's `rerank` model re-scores those 20
   by actually reading the query and each candidate together, keeping the
   best 5. Two stages exist because rerank is far more accurate but too
   expensive to run over the entire document set on every question.
4. **Generate** (`src/generate.py`) — the top 5 chunks are passed to Cohere's
   `chat` endpoint (Command A) as structured grounding documents. The model
   is constrained to answer from those documents specifically, and returns
   citations linking claims back to source documents, which are resolved back
   to real filenames.

## Tech stack

- Python 3.13
- [`cohere`](https://pypi.org/project/cohere/) — embed, rerank, and chat (Command A)
- `pypdf` — PDF text extraction
- `numpy` — in-memory vector storage and cosine similarity search
- `python-dotenv` — API key management

## Setup

1. **Clone and create a virtual environment:**
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. **Add your Cohere API key** (free trial keys work fine — get one at
   [dashboard.cohere.com](https://dashboard.cohere.com)):
   ```bash
   cp .env.example .env
   # then edit .env and paste in your real key
   ```

3. **Download the source documents** — see [`data/SOURCES.md`](data/SOURCES.md)
   for direct links to the official HSE PDFs used, and save them into
   `data/pdfs/`.

4. **Run it:**
   ```bash
   .venv/bin/python main.py
   ```
   The first run embeds all document chunks via the Cohere API (a few
   seconds); subsequent runs load instantly from a local cache. Ask a
   question, or type `exit` to quit.

## Testing

Unit tests cover chunking behavior (`src/loader.py`), retrieval (`src/retrieval.py`),
and grounded generation (`src/generate.py`). All Cohere API calls are mocked,
so the suite runs instantly and doesn't need a real API key or network access.

Install test dependencies and run the suite:
```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

## Running with Docker

Build the image:
```bash
docker build -t cohere-construction-qa .
```

Run it, passing your API key at runtime rather than baking it into the image,
and mounting your local `data/pdfs/` folder (also excluded from the image
via `.dockerignore`, for the same reason — source documents shouldn't be
baked in either):
```bash
docker run -it --rm \
  -e COHERE_API_KEY=your_cohere_api_key_here \
  -v "$(pwd)/data/pdfs:/app/data/pdfs" \
  -v "$(pwd)/data/cache:/app/data/cache" \
  cohere-construction-qa
```
The second volume (`data/cache`) is optional — without it, each container
run re-embeds every chunk from scratch instead of reusing the on-disk cache.

## Design notes

A few decisions worth calling out:

- **In-memory + disk-cached vectors, not a vector database.** At this scale
  (hundreds of chunks), a NumPy matrix and a normalized dot product is simpler,
  faster to set up, and fully sufficient — no need for the operational
  overhead of Pinecone/Weaviate/etc. for a project like this.
- **Retrieve-then-rerank rather than vector search alone.** Vector search is
  fast but approximate; rerank is precise but expensive. Using both in
  sequence gets the accuracy benefit of reranking without paying its cost
  across the entire document set.
- **Citations, not just answers.** Cohere's chat endpoint tracks which input
  documents actually supported each generated claim — this is surfaced back
  to the user, so answers are verifiable against real source text rather than
  trusted blindly.
