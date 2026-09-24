from pathlib import Path

from config.settings import SETTINGS

from .embeddings import BGEEmbedder
from .index import HybridIndex
from .reranker import BGEReranker
from .llm import get_llm, SYSTEM_PROMPT


class RAGPipeline:
    def __init__(
        self,
        index,
        reranker,
        llm,
    ):
        self.index = index
        self.reranker = reranker
        self.llm = llm

    # ---------------------------------------------------------
    # Load pipeline from disk
    # ---------------------------------------------------------
    @classmethod
    def from_disk(cls):
        required = [
            SETTINGS.index_dir / "dense.faiss",
            SETTINGS.index_dir / "chunks.json",
            SETTINGS.index_dir / "bm25.pkl",
        ]

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Index not found. Run: "
                "python -m scripts.ingest"
            )

        embedder = BGEEmbedder(
            SETTINGS.embedding_model
        )

        index = HybridIndex.load(
            SETTINGS.index_dir,
            embedder,
            SETTINGS.rrf_k,
        )

        reranker = BGEReranker(
            SETTINGS.reranker_model
        )

        llm = get_llm()

        return cls(
            index,
            reranker,
            llm,
        )

    # ---------------------------------------------------------
    # Retrieve relevant chunks
    # ---------------------------------------------------------
    def retrieve(
        self,
        question,
        document=None,
    ):
        """
        Retrieve chunks using hybrid dense + BM25 retrieval.

        If document is supplied, retrieval is restricted to
        that contract.
        """

        candidates = self.index.hybrid_search(
            question,
            SETTINGS.dense_top_k,
            SETTINGS.bm25_top_k,
            document=document,
        )

        # If a target document was requested but no chunks
        # were found, return an empty result instead of using
        # another document.
        if document and not candidates:
            return []

        return self.reranker.rerank(
            question,
            candidates,
            SETTINGS.rerank_top_k,
        )

    # ---------------------------------------------------------
    # Build LLM context
    # ---------------------------------------------------------
    def build_context(
        self,
        reranked,
    ):
        blocks = []

        for i, item in enumerate(
            reranked,
            start=1,
        ):
            c = item["chunk"]

            blocks.append(
                f"[SOURCE {i}]\n"
                f"Document: {c.document}\n"
                f"Section: "
                f"{c.section_id} "
                f"{c.section_title}\n"
                f"Parent section: "
                f"{c.parent_section}\n"
                f"Pages: "
                f"{c.page_start}-{c.page_end}\n"
                f"Content:\n"
                f"{c.text}"
            )

        return "\n\n".join(
            blocks
        )

    # ---------------------------------------------------------
    # Generate answer
    # ---------------------------------------------------------
    def answer(
        self,
        question,
        document=None,
    ):
        """
        Generate a grounded answer.

        Parameters
        ----------
        question:
            User question.

        document:
            Optional target document.

            When supplied, retrieval is strictly restricted
            to that document.
        """

        reranked = self.retrieve(
            question,
            document=document,
        )

        # -----------------------------------------------------
        # No target document found
        # -----------------------------------------------------
        if document and not reranked:
            return {
                "question": question,
                "answer": (
                    f"No indexed chunks were found for the "
                    f"target document '{document}'."
                ),
                "sources": [],
            }

        # -----------------------------------------------------
        # Build context
        # -----------------------------------------------------
        context = self.build_context(
            reranked
        )

        # -----------------------------------------------------
        # Build prompt
        # -----------------------------------------------------
        if document:
            document_instruction = (
                f"\nTARGET DOCUMENT:\n"
                f"{document}\n\n"
                f"IMPORTANT:\n"
                f"Answer the question using only information "
                f"from the target document above. Do not use "
                f"information from other contracts or documents."
            )
        else:
            document_instruction = ""

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{document_instruction}\n\n"
            f"RETRIEVED CONTEXT:\n"
            f"{context}\n\n"
            f"USER QUESTION:\n"
            f"{question}\n\n"
            f"Return the grounded answer and cite the "
            f"source sections/pages."
        )

        # -----------------------------------------------------
        # LLM generation
        # -----------------------------------------------------
        response = self.llm.complete(
            prompt
        )

        # -----------------------------------------------------
        # Build source metadata
        # -----------------------------------------------------
        sources = []

        for item in reranked:
            c = item["chunk"]

            sources.append(
                {
                    "document": c.document,
                    "section": c.section_id,
                    "section_title": c.section_title,
                    "pages": (
                        f"{c.page_start}-"
                        f"{c.page_end}"
                    ),
                    "retrieval_score": round(
                        item["retrieval_score"],
                        6,
                    ),
                    "rerank_score": round(
                        item["rerank_score"],
                        6,
                    ),
                    "text": c.text,
                }
            )

        return {
            "question": question,
            "answer": str(response),
            "sources": sources,
        }