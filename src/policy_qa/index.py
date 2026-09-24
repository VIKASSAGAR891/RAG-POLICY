import json
import pickle
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from .models import Chunk


class HybridIndex:
    def __init__(self, index_dir: Path, embedder, rrf_k=60):
        self.index_dir = Path(index_dir)
        self.embedder = embedder
        self.rrf_k = rrf_k

        self.faiss_index = None
        self.chunks = []
        self.bm25 = None

    # ---------------------------------------------------------
    # Build index
    # ---------------------------------------------------------
    def build(self, chunks):
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.chunks = chunks

        # Dense embeddings
        vectors = self.embedder.encode_documents(
            [c.text for c in chunks]
        )

        self.faiss_index = faiss.IndexFlatIP(
            vectors.shape[1]
        )

        self.faiss_index.add(vectors)

        # BM25 sparse index
        tokenized = [
            c.text.lower().split()
            for c in chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

        # Save FAISS
        faiss.write_index(
            self.faiss_index,
            str(self.index_dir / "dense.faiss"),
        )

        # Save chunks
        with open(
            self.index_dir / "chunks.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                [c.to_dict() for c in chunks],
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Save BM25
        with open(
            self.index_dir / "bm25.pkl",
            "wb",
        ) as f:
            pickle.dump(self.bm25, f)

        return len(chunks)

    # ---------------------------------------------------------
    # Load existing index
    # ---------------------------------------------------------
    @classmethod
    def load(
        cls,
        index_dir,
        embedder,
        rrf_k=60,
    ):
        obj = cls(
            index_dir,
            embedder,
            rrf_k,
        )

        index_dir = Path(index_dir)

        obj.faiss_index = faiss.read_index(
            str(index_dir / "dense.faiss")
        )

        with open(
            index_dir / "chunks.json",
            encoding="utf-8",
        ) as f:
            obj.chunks = [
                Chunk(**x)
                for x in json.load(f)
            ]

        with open(
            index_dir / "bm25.pkl",
            "rb",
        ) as f:
            obj.bm25 = pickle.load(f)

        return obj

    # ---------------------------------------------------------
    # Normalize document names
    # ---------------------------------------------------------
    @staticmethod
    def normalize_document_name(document):
        """
        Normalizes a document name so that:

            Example.pdf
            Example

        are treated as the same document.

        Case differences are also ignored.
        """

        if document is None:
            return ""

        value = str(document).strip()

        if not value:
            return ""

        if value.casefold().endswith(".pdf"):
            value = value[:-4]

        return value.strip().casefold()

    # ---------------------------------------------------------
    # Find chunk IDs belonging to a document
    # ---------------------------------------------------------
    def get_document_indices(self, document):
        """
        Returns the FAISS/BM25 chunk indices belonging
        to the requested document.

        CUAD document names may not contain .pdf while
        indexed documents normally do, so normalization
        is applied before comparison.
        """

        target = self.normalize_document_name(
            document
        )

        if not target:
            return []

        matching_indices = []

        for idx, chunk in enumerate(self.chunks):
            chunk_document = self.normalize_document_name(
                chunk.document
            )

            if chunk_document == target:
                matching_indices.append(idx)

        return matching_indices

    # ---------------------------------------------------------
    # Dense retrieval
    # ---------------------------------------------------------
    def dense_search(
        self,
        query,
        top_k,
        allowed_indices=None,
    ):
        """
        Dense FAISS retrieval.

        If allowed_indices is supplied, retrieval is restricted
        to those chunk IDs.
        """

        q = self.embedder.encode_query(
            query
        ).reshape(1, -1)

        # -----------------------------------------------------
        # Global retrieval
        # -----------------------------------------------------
        if allowed_indices is None:
            search_k = min(
                top_k,
                len(self.chunks),
            )

            scores, ids = self.faiss_index.search(
                q,
                search_k,
            )

            return [
                (int(idx), float(score))
                for idx, score in zip(
                    ids[0],
                    scores[0],
                )
                if idx >= 0
            ]

        # -----------------------------------------------------
        # Document-filtered retrieval
        # -----------------------------------------------------
        if not allowed_indices:
            return []

        # Reconstruct only the vectors belonging to the
        # requested document.
        vectors = []

        valid_indices = []

        for idx in allowed_indices:
            try:
                vector = self.faiss_index.reconstruct(
                    int(idx)
                )

                vectors.append(vector)
                valid_indices.append(idx)

            except Exception:
                continue

        if not vectors:
            return []

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        # Since the FAISS index uses inner product and the
        # embeddings are normalized, dot product gives
        # cosine similarity.
        scores = np.dot(
            vectors,
            q[0],
        )

        order = np.argsort(
            scores
        )[::-1]

        order = order[
            : min(
                top_k,
                len(order),
            )
        ]

        return [
            (
                int(valid_indices[i]),
                float(scores[i]),
            )
            for i in order
        ]

    # ---------------------------------------------------------
    # BM25 retrieval
    # ---------------------------------------------------------
    def bm25_search(
        self,
        query,
        top_k,
        allowed_indices=None,
    ):
        """
        BM25 sparse retrieval.

        If allowed_indices is supplied, only chunks belonging
        to the specified document are considered.
        """

        scores = self.bm25.get_scores(
            query.lower().split()
        )

        # -----------------------------------------------------
        # Global BM25
        # -----------------------------------------------------
        if allowed_indices is None:
            ids = np.argsort(
                scores
            )[::-1][
                : min(
                    top_k,
                    len(scores),
                )
            ]

            return [
                (int(idx), float(scores[idx]))
                for idx in ids
                if scores[idx] > 0
            ]

        # -----------------------------------------------------
        # Document-filtered BM25
        # -----------------------------------------------------
        if not allowed_indices:
            return []

        allowed_scores = [
            (
                int(idx),
                float(scores[idx]),
            )
            for idx in allowed_indices
            if scores[idx] > 0
        ]

        allowed_scores.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return allowed_scores[
            : min(
                top_k,
                len(allowed_scores),
            )
        ]

    # ---------------------------------------------------------
    # Hybrid RRF retrieval
    # ---------------------------------------------------------
    def hybrid_search(
        self,
        query,
        dense_k,
        bm25_k,
        document=None,
    ):
        """
        Hybrid retrieval using:

            Dense FAISS
                  +
            BM25
                  ↓
            Reciprocal Rank Fusion

        If document is supplied, BOTH retrieval methods are
        restricted to that document before RRF is performed.
        """

        # -----------------------------------------------------
        # Determine document filter
        # -----------------------------------------------------
        if document:
            allowed_indices = self.get_document_indices(
                document
            )

            # If a document was explicitly requested but
            # cannot be found, return no results instead of
            # accidentally retrieving from another contract.
            if not allowed_indices:
                return []

        else:
            allowed_indices = None

        # -----------------------------------------------------
        # Dense retrieval
        # -----------------------------------------------------
        dense = self.dense_search(
            query,
            dense_k,
            allowed_indices=allowed_indices,
        )

        # -----------------------------------------------------
        # BM25 retrieval
        # -----------------------------------------------------
        sparse = self.bm25_search(
            query,
            bm25_k,
            allowed_indices=allowed_indices,
        )

        # -----------------------------------------------------
        # Reciprocal Rank Fusion
        # -----------------------------------------------------
        ranks = {}

        for rank, (idx, _) in enumerate(
            dense,
            start=1,
        ):
            ranks[idx] = (
                ranks.get(idx, 0.0)
                + 1.0
                / (
                    self.rrf_k + rank
                )
            )

        for rank, (idx, _) in enumerate(
            sparse,
            start=1,
        ):
            ranks[idx] = (
                ranks.get(idx, 0.0)
                + 1.0
                / (
                    self.rrf_k + rank
                )
            )

        ordered = sorted(
            ranks.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            (
                self.chunks[idx],
                score,
            )
            for idx, score in ordered
        ]