import streamlit as st
from .pipeline import RAGPipeline

@st.cache_resource
def load_pipeline():
    return RAGPipeline.from_disk()

def run_streamlit():
    st.set_page_config(page_title="Policy Document QA", layout="wide")
    st.title("Policy Document QA & Extraction Engine")
    st.caption("Evaluated RAG: structural chunking + hybrid retrieval + BGE reranking + grounded generation")

    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error(str(exc))
        st.info("Run `python scripts/ingest.py` after adding PDFs to data/raw/ and configure GROQ_API_KEY in .env.")
        return

    question = st.text_area(
        "Ask a question about the indexed documents",
        placeholder="e.g. Under what circumstances can the agreement be terminated?",
        height=120,
    )

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving, reranking and generating..."):
            result = pipeline.answer(question)
        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Retrieved evidence")
        for i, source in enumerate(result["sources"], start=1):
            title = f"{i}. {source['document']} — Section {source['section']} — Pages {source['pages']}"
            with st.expander(title):
                st.caption(
                    f"Retrieval score: {source['retrieval_score']} | "
                    f"Rerank score: {source['rerank_score']}"
                )
                st.write(source["text"])
