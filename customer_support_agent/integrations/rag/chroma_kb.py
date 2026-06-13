from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, Embeddings, Space
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from customer_support_agent.core.settings import Settings


class GoogleGenaiOneByOneEmbeddingFunction:
    """Embed each document separately so Chroma receives one vector per chunk."""

    def __init__(self, api_key: str, model_name: str):
        try:
            import google.genai as genai
        except ImportError as exc:
            raise RuntimeError("Install `google-genai` to use Gemini embeddings.") from exc

        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            raise ValueError("Input documents cannot be empty")

        return self.embed_documents(list(input))

    def embed_documents(self, texts: list[str] | None = None, **kwargs: Any) -> list[list[float]]:
        texts = texts or kwargs.get("input") or []
        embeddings: list[list[float]] = []
        for text in texts:
            embeddings.append(self._embed_one(text))

        return embeddings

    def embed_query(self, text: str | list[str] | None = None, **kwargs: Any) -> list[list[float]]:
        query_input = text or kwargs.get("input")
        if not query_input:
            raise ValueError("Input query cannot be empty")
        if isinstance(query_input, (list, tuple)):
            return self.embed_documents(list(query_input))
        return [self._embed_one(query_input)]

    def _embed_one(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        if not response.embeddings:
            raise ValueError("No embedding returned from Gemini")
        return response.embeddings[0].values

    @staticmethod
    def name() -> str:
        return "google_genai_one_by_one"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> list[Space]:
        return ["cosine", "l2", "ip"]


class KnowledgeBaseService:
    def __init__(self, settings:Settings):
        self._settings = settings
        self._client = chromadb.PersistentClient(path=str(settings.chroma_rag_path))
        self._collection_name = "support_kb_gemini" if settings.google_api_key else "support_kb"
        self._embedding_function = self._build_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

    def _build_embedding_function(self) -> Any:
        if self._settings.google_api_key:
            os.environ.setdefault("GOOGLE_API_KEY", self._settings.google_api_key)
            os.environ.setdefault("GEMINI_API_KEY", self._settings.google_api_key)
            try:
                return GoogleGenaiOneByOneEmbeddingFunction(
                    api_key=self._settings.google_api_key,
                    model_name=self._settings.effective_google_embedding_model,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Gemini embedding initialization failed. Install `google-genai` and verify GOOGLE_API_KEY."
                ) from exc

        return embedding_functions.DefaultEmbeddingFunction()

    def ingest_directory(self, directory: Path, clear_existing: bool = False) -> dict[str, int]:
        if clear_existing:
            self._client.delete_collection(name=self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=self._embedding_function,
            )
        
        source_files = sorted(
            [
                *directory.glob("*.md"),
                *directory.glob("*.txt"),
            ]
        )

        docs: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for file_path in source_files:
            text = file_path.read_text(encoding="utf-8")
            chunks = self._splitter.split_text(text)

            for index, chunk in enumerate(chunks):
                chunk_hash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:10]
                doc_id = f"{file_path.stem}-{index}-{chunk_hash}"
                docs.append(chunk)
                ids.append(doc_id)
                metadatas.append(
                    {
                        "source": file_path.name,
                        "chunk_index": index,
                    }
                )
            
        if docs:
            # upsert prevents duplicate-id failures when re-ingesting.
            self._collection.upsert(documents=docs, ids=ids, metadatas=metadatas)

        return {
            "files_indexed": len(source_files),
            "chunks_indexed": len(docs),
            "collection_count": self._collection.count(),
        }

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k or self._settings.rag_top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        combined: list[dict[str, Any]] = []
        for i, document in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else None
            combined.append(
                {
                    "content": document,
                    "source": metadata.get("source", "unknown"),
                    "distance": distance,
                }
            )

        return combined
