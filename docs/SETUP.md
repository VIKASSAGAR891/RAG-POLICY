# Setup checklist

1. Install Python 3.10+.
2. Create and activate `.venv`.
3. Install `requirements.txt`.
4. Copy `.env.example` to `.env`.
5. Add `GROQ_API_KEY`.
6. Put PDFs under `data/raw/`.
7. Run `python scripts/ingest.py`.
8. Run `python scripts/query.py "your question"`.
9. Run `streamlit run app.py` for the UI.
10. Create `data/evaluation/questions.jsonl`.
11. Run `python scripts/evaluate.py`.

The first ingestion can take time because the embedding model is downloaded and all PDFs are processed.
The BGE reranker is also downloaded the first time a query is executed.
