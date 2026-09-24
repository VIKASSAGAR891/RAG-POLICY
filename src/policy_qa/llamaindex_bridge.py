"""
LlamaIndex bridge.

The core retrieval logic remains explicit (BM25 + FAISS + BGE reranking), while
LlamaIndex is used for the LLM integration. This adapter also exposes retrieved
chunks as LlamaIndex TextNodes for future QueryEngine/tool integrations.
"""

from llama_index.core.schema import TextNode
from .models import Chunk


def chunks_to_llama_nodes(chunks: list[Chunk]):
    nodes = []
    for chunk in chunks:
        nodes.append(
            TextNode(
                id_=chunk.chunk_id,
                text=chunk.text,
                metadata={
                    "document": chunk.document,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                    "parent_section": chunk.parent_section,
                },
            )
        )
    return nodes
