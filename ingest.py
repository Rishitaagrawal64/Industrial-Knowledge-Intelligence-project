"""
Step 1: Ingestion
Loads PDFs/text files, splits them into chunks, embeds them,
and stores them in a local Chroma vector database.
"""

import os
import pdfplumber
import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "industrial_docs"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap so context isn't cut mid-sentence


def load_pdf_text(filepath):
    """Extract text page by page from a PDF, keeping page numbers."""
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text})
    return pages


def load_text_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return [{"page": 1, "text": f.read()}]


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Chroma's default embedding function runs locally (no API key needed)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def ingest_file(filepath, collection):
    filename = os.path.basename(filepath)
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        pages = load_pdf_text(filepath)
    elif ext in ("txt", "md"):
        pages = load_text_file(filepath)
    else:
        print(f"Skipping unsupported file: {filename}")
        return 0

    doc_id_counter = 0
    ids, docs, metadatas = [], [], []

    for page in pages:
        for chunk in chunk_text(page["text"]):
            if not chunk.strip():
                continue
            doc_id_counter += 1
            chunk_id = f"{filename}::p{page['page']}::c{doc_id_counter}"
            ids.append(chunk_id)
            docs.append(chunk)
            metadatas.append({
                "source": filename,
                "page": page["page"],
            })

    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas)

    return len(ids)


def ingest_folder(folder_path):
    collection = get_chroma_collection()
    total_chunks = 0
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            n = ingest_file(fpath, collection)
            print(f"Ingested {fname}: {n} chunks")
            total_chunks += n
    print(f"Total chunks ingested: {total_chunks}")
    return collection


if __name__ == "__main__":
    ingest_folder("./sample_docs")
