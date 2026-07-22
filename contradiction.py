"""
Step 4: The differentiator.
Given a piece of equipment (or topic), pull together everything tagged
with it across document types (procedure, standard, incident report)
and ask the LLM specifically to find contradictions / compliance gaps /
repeated-failure patterns between them.
"""

import anthropic
from ingest import get_chroma_collection

client = anthropic.Anthropic()

GAP_PROMPT = """You are a safety and compliance auditor reviewing documents
for a single piece of equipment or topic: "{topic}"

Below are excerpts from different document types (procedures, standards,
incident reports, inspection reports) related to this topic. Your job:

1. Identify any CONTRADICTIONS between what the procedure says and what
   the standard requires.
2. Identify any PAST INCIDENT that is linked to a gap still present in
   current procedures.
3. If nothing meaningful is found, say so honestly.

Be specific and cite [Source: filename, page X] for every claim.

Document excerpts:
{context}

Respond in this format:
FINDING: <clear description of the contradiction/gap, or "No significant gaps found">
RISK_LEVEL: <High / Medium / Low / None>
EVIDENCE: <cited excerpts supporting the finding>
RECOMMENDATION: <what should be done about it>
"""


def find_related_chunks(topic, collection, n_results=8):
    """Retrieve chunks related to a topic/equipment across all doc types."""
    results = collection.query(query_texts=[topic], n_results=n_results)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "doc_type": meta.get("doc_type", "other"),
        })
    return chunks


def detect_gaps(topic, collection=None):
    if collection is None:
        collection = get_chroma_collection()

    chunks = find_related_chunks(topic, collection)
    if not chunks:
        return {"finding": "No documents found for this topic.", "risk_level": "None"}

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, page {c['page']}, type: {c['doc_type']}]\n{c['text']}"
        for c in chunks
    )
    prompt = GAP_PROMPT.format(topic=topic, context=context)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text

    def extract_field(name, text):
        marker = f"{name}:"
        if marker not in text:
            return ""
        after = text.split(marker, 1)[1]
        for other in ["FINDING:", "RISK_LEVEL:", "EVIDENCE:", "RECOMMENDATION:"]:
            if other != marker and other in after:
                after = after.split(other)[0]
        return after.strip()

    return {
        "finding": extract_field("FINDING", raw),
        "risk_level": extract_field("RISK_LEVEL", raw),
        "evidence": extract_field("EVIDENCE", raw),
        "recommendation": extract_field("RECOMMENDATION", raw),
    }


if __name__ == "__main__":
    coll = get_chroma_collection()
    result = detect_gaps("Pump-104 maintenance procedure", coll)
    for k, v in result.items():
        print(f"\n{k.upper()}:\n{v}")
