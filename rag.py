"""
Step 3: RAG — retrieve relevant chunks, generate a cited answer.
"""

import anthropic
from ingest import get_chroma_collection

client = anthropic.Anthropic()

ANSWER_PROMPT = """You are an industrial knowledge assistant. Answer the
question using ONLY the provided document excerpts. Every claim must be
backed by a source. If the excerpts don't contain enough information,
say so honestly instead of guessing.

Question: {question}

Document excerpts:
{context}

Respond in this exact format:
ANSWER: <your answer, with inline citations like [Source: filename, page X]>
CONFIDENCE: <High / Medium / Low>
"""


def retrieve(question, collection, n_results=5):
    results = collection.query(query_texts=[question], n_results=n_results)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist})
    return chunks


def build_context(chunks):
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']}, page {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(question, collection=None):
    if collection is None:
        collection = get_chroma_collection()

    chunks = retrieve(question, collection)
    if not chunks:
        return {"answer": "No relevant documents found.", "confidence": "Low", "sources": []}

    context = build_context(chunks)
    prompt = ANSWER_PROMPT.format(question=question, context=context)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text

    answer_part = raw.split("CONFIDENCE:")[0].replace("ANSWER:", "").strip()
    confidence_part = raw.split("CONFIDENCE:")[1].strip() if "CONFIDENCE:" in raw else "Medium"

    sources = list({f"{c['source']} (p.{c['page']})" for c in chunks})

    return {"answer": answer_part, "confidence": confidence_part, "sources": sources}


if __name__ == "__main__":
    coll = get_chroma_collection()
    result = answer_question("What maintenance issues have occurred on Pump-104?", coll)
    print(result)
