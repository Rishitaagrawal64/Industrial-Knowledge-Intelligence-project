🏭 Industrial Knowledge Intelligence

A unified AI layer over fragmented industrial documents — maintenance reports, safety procedures, regulatory standards, and incident reports — that answers questions with cited sources and surfaces the compliance gaps hiding between them.

Built on Retrieval-Augmented Generation (RAG) with a lightweight LLM-driven entity tagging layer standing in for a full knowledge graph.

Why

Industrial knowledge lives in scattered PDFs and folders. A procedure can quietly drift out of step with the standard it's meant to satisfy, or repeat a mistake a past incident report already flagged — and nobody notices until an audit or, worse, another incident. This project explores whether an LLM can be pointed at that raw document set and (1) answer operational questions with verifiable citations, and (2) proactively flag contradictions across document types.

How it works
Documents (PDF/TXT/MD)
        │
        ▼
  Chunk + Embed  ──────────────►  Chroma vector store (local, on disk)
        │
        ▼
  Entity Tagging (Claude)
  equipment · doc_type · standards · dates
        │
        ├──────────────► Ask a Question (RAG)
        │                 retrieve → cite → confidence
        │
        └──────────────► Find Compliance Gaps
                          cross-reference procedures / standards / incidents
Ingest (ingest.py) — PDFs and text files are split into overlapping 800-character chunks and embedded into a local Chroma vector store. Nothing leaves the machine.
Extract & Tag (extract.py) — each chunk is sent to Claude to extract structured metadata (equipment IDs, document type, referenced standards, dates). Chunks that share tags are effectively "linked," giving graph-like retrieval without a graph database.
Ask (rag.py) — a question is embedded, matched against the most relevant chunks, and answered with inline [Source: filename, page X] citations and a High/Medium/Low confidence rating. If the documents don't say enough, it says so instead of guessing.
Audit (contradiction.py) — given a topic (e.g. Pump-104 or confined space entry), pulls together everything tagged with it across procedures, standards, and incident reports, and asks Claude specifically to find:
Contradictions between what a procedure says and what a standard requires
Recurring risk — a past incident linked to a gap still present today

A three-tab Streamlit UI (app.py) ties it together: 📂 Upload Documents · 💬 Ask a Question · ⚠️ Find Compliance Gaps.

Project structure
.
├── app.py              # Streamlit demo UI (3 tabs)
├── ingest.py            # PDF/text loading, chunking, embedding, Chroma storage
├── extract.py            # LLM entity extraction / metadata tagging
├── rag.py               # Retrieval-augmented Q&A with citations
├── contradiction.py     # Cross-document contradiction / compliance gap detection
├── requirements.txt
└── sample_docs/         # Drop sample PDFs/text files here
Setup
bash
git clone <this-repo>
cd knowledge-intel
python -m venv venv && source venv/bin/activate    # optional but recommended
pip install -r requirements.txt

Set your Anthropic API key:

bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
Usage
Run the demo UI
bash
streamlit run app.py

Then in the browser:

Upload Documents — upload PDFs/text files and click Ingest & Tag.
Ask a Question — e.g. "What maintenance issues have occurred on Pump-104 in the last year?"
Find Compliance Gaps — enter an equipment ID or topic and click Run Audit.
Or run pieces from the command line
bash
# Ingest everything in ./sample_docs
python ingest.py

# Tag all ingested chunks with extracted entities
python extract.py

# Ask a question
python rag.py

# Run a compliance audit on a topic
python contradiction.py
Tech stack
Component	Role
Streamlit	Demo UI
pdfplumber	Page-level PDF text extraction
ChromaDB	Local vector store (default embedding function, no API key needed)
Anthropic Claude	Entity tagging, RAG answers, contradiction detection
python-dotenv	Loads ANTHROPIC_API_KEY from .env
Notes & limitations
This is a prototype: entity tagging happens per-chunk with one LLM call each, so tagging a large corpus will be slow and will consume API credits.
The vector store persists to ./chroma_db; delete that folder to start fresh.
Chroma's default embedding function runs locally — no embedding API key required.
sample_docs/ ships empty — add your own PDFs/text files to try it out.
Roadmap ideas
 Swap the sliding-window chunker for a structure-aware splitter (headings, tables)
 Batch entity extraction instead of one call per chunk
 Persist and surface the "knowledge graph" (shared tags) visually
 Add authentication before deploying beyond a local demo
