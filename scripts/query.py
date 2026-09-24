import sys
from src.policy_qa.pipeline import RAGPipeline

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python scripts/query.py "your question"')
    pipeline = RAGPipeline.from_disk()
    result = pipeline.answer(question)
    print("\nANSWER\n-------")
    print(result["answer"])
    print("\nSOURCES\n-------")
    for i, source in enumerate(result["sources"], 1):
        print(
            f"[{i}] {source['document']} | Section {source['section']} "
            f"| Pages {source['pages']} | rerank={source['rerank_score']}"
        )
