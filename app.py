"""
Step 5: The demo UI.
Three tabs: Upload & Ingest | Ask a Question | Find Compliance Gaps
"""

import os
import streamlit as st

from ingest import get_chroma_collection, ingest_file
from extract import tag_collection
from rag import answer_question
from contradiction import detect_gaps

st.set_page_config(page_title="Industrial Knowledge Intelligence", layout="wide")
st.title("🏭 Industrial Knowledge Intelligence")
st.caption("Unified AI layer over fragmented industrial documents")

collection = get_chroma_collection()

tab1, tab2, tab3 = st.tabs(["📂 Upload Documents", "💬 Ask a Question", "⚠️ Find Compliance Gaps"])

with tab1:
    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files", accept_multiple_files=True, type=["pdf", "txt", "md"]
    )
    if uploaded_files and st.button("Ingest & Tag"):
        os.makedirs("./sample_docs", exist_ok=True)
        with st.spinner("Ingesting documents..."):
            for f in uploaded_files:
                path = os.path.join("./sample_docs", f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                n = ingest_file(path, collection)
                st.write(f"✅ {f.name}: {n} chunks ingested")
        with st.spinner("Extracting entities and tagging (this calls the LLM per chunk)..."):
            tag_collection(collection)
        st.success("Done! You can now ask questions or check for compliance gaps.")

with tab2:
    st.subheader("Ask a question across all documents")
    question = st.text_input(
        "e.g. What maintenance issues have occurred on Pump-104 in the last year?"
    )
    if st.button("Ask") and question:
        with st.spinner("Retrieving and reasoning..."):
            result = answer_question(question, collection)
        st.markdown(f"**Answer:** {result['answer']}")
        st.markdown(f"**Confidence:** {result['confidence']}")
        with st.expander("Sources"):
            for s in result["sources"]:
                st.write(f"- {s}")

with tab3:
    st.subheader("Find contradictions & compliance gaps")
    st.caption(
        "Checks whether procedures conflict with standards, or whether "
        "past incidents are linked to gaps still present today."
    )
    topic = st.text_input("Equipment or topic to audit, e.g. 'Pump-104' or 'confined space entry'")
    if st.button("Run Audit") and topic:
        with st.spinner("Cross-referencing procedures, standards, and incident history..."):
            result = detect_gaps(topic, collection)

        risk = result.get("risk_level", "").strip()
        color = {"High": "🔴", "Medium": "🟠", "Low": "🟡", "None": "🟢"}.get(risk, "⚪")

        st.markdown(f"### {color} Risk Level: {risk}")
        st.markdown(f"**Finding:** {result.get('finding')}")
        st.markdown(f"**Evidence:** {result.get('evidence')}")
        st.markdown(f"**Recommendation:** {result.get('recommendation')}")
