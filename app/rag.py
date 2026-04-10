from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.md"


def _load_documents() -> list[Document]:
    content = KB_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["## ", "\n### ", "\n\n", "\n", " "],
    )
    return splitter.create_documents([content])


@lru_cache(maxsize=1)
def get_retriever() -> BM25Retriever:
    retriever = BM25Retriever.from_documents(_load_documents())
    retriever.k = 3
    return retriever


def retrieve_context(query: str) -> str:
    docs = get_retriever().invoke(query)
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)
