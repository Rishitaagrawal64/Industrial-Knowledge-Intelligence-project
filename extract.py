"""
Step 2: Entity Extraction (fake "knowledge graph" via metadata tags)
Instead of building a full graph database, we tag each chunk with
structured entities using an LLM call. Chunks sharing tags = "linked"
in the graph, without needing Neo4j.
"""

import json
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

EXTRACTION_PROMPT = """You are tagging a chunk of an industrial document.
Extract structured entities from the text below. Return ONLY valid JSON,
no preamble, no markdown fences.

Schema:
{{
  "equipment": ["list of equipment names/tags mentioned, e.g. Pump-104"],
  "doc_type": "one of: maintenance_report | safety_procedure | inspection_report | incident_report | regulatory_standard | other",
  "standards_referenced": ["list of standards/codes mentioned, e.g. OISD-105"],
  "dates": ["list of dates mentioned, ISO format if possible"]
}}

Text:
\"\"\"{text}\"\"\"
"""


def extract_entities(chunk_text):
    prompt = EXTRACTION_PROMPT.format(text=chunk_text[:3000])
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"equipment": [], "doc_type": "other", "standards_referenced": [], "dates": []}


def tag_collection(collection, batch_size=20):
    """Pull all chunks from Chroma, extract entities, write tags back as metadata."""
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    docs = all_data["documents"]
    metas = all_data["metadatas"]

    updated_metas = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        entities = extract_entities(doc)
        meta["equipment"] = ",".join(entities.get("equipment", []))
        meta["doc_type"] = entities.get("doc_type", "other")
        meta["standards"] = ",".join(entities.get("standards_referenced", []))
        updated_metas.append(meta)
        if (i + 1) % 5 == 0:
            print(f"Tagged {i+1}/{len(ids)} chunks")

    collection.update(ids=ids, metadatas=updated_metas)
    print("Tagging complete.")


if __name__ == "__main__":
    from ingest import get_chroma_collection
    coll = get_chroma_collection()
    tag_collection(coll)
