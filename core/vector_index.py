"""ChromaDB and SentenceTransformer vector index for Obsidian vault paragraphs."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer

from core.models import VaultParagraph

logger = logging.getLogger(__name__)


class VaultVectorIndex:
    """Manages local embedding generation and ChromaDB indexing for vault paragraphs."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "vault_paragraphs",
    ):
        self.model_name = model_name
        self.collection_name = collection_name
        self._model: Optional[SentenceTransformer] = None
        self._client = chromadb.Client()
        self._collection = None
        self._paragraphs: List[VaultParagraph] = []

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer to optimize startup time."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def reset_collection(self) -> None:
        """Reset or initialize the Chroma collection with cosine distance metric."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass

        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._paragraphs = []

    def index_paragraphs(self, paragraphs: List[VaultParagraph]) -> int:
        """Encode and index paragraphs into ChromaDB.

        Returns number of indexed paragraphs.
        """
        self.reset_collection()
        if not paragraphs:
            return 0

        self._paragraphs = paragraphs
        texts = [p.text for p in paragraphs]

        # Batch encode with normalized embeddings
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        ids = [f"p_{i}" for i in range(len(paragraphs))]
        metadatas = [
            {
                "file_path": p.file_path,
                "rel_path": p.rel_path,
                "note_title": p.note_title,
                "section_heading": p.section_heading or "",
                "paragraph_index": p.paragraph_index,
            }
            for p in paragraphs
        ]

        # Chroma requires non-empty string in metadata values
        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts,
        )

        return len(paragraphs)

    def query_most_similar(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> Tuple[float, Optional[VaultParagraph], List[Tuple[float, VaultParagraph]]]:
        """Query vector index for paragraphs most similar to query_text.

        Returns:
            (max_similarity, best_matching_paragraph, list_of_top_matches)
        """
        if self._collection is None or self._collection.count() == 0:
            return 0.0, None, []

        query_emb = self.model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        k = min(top_k, self._collection.count())

        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["distances"] or not results["distances"][0]:
            return 0.0, None, []

        matches: List[Tuple[float, VaultParagraph]] = []
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for dist, doc, meta in zip(distances, documents, metadatas):
            # Chroma cosine distance: dist = 1.0 - cosine_similarity
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            paragraph = VaultParagraph(
                file_path=meta.get("file_path", ""),
                rel_path=meta.get("rel_path", ""),
                note_title=meta.get("note_title", ""),
                section_heading=meta.get("section_heading") or None,
                paragraph_index=int(meta.get("paragraph_index", 0)),
                text=doc,
            )
            matches.append((similarity, paragraph))

        # Highest similarity is the first result
        max_sim, best_match = matches[0]
        return max_sim, best_match, matches

    def add_single_paragraph(self, paragraph: VaultParagraph) -> None:
        """Add a newly written paragraph to the in-memory index."""
        if self._collection is None:
            self.reset_collection()

        self._paragraphs.append(paragraph)
        emb = self.model.encode([paragraph.text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        doc_id = f"p_{self._collection.count()}"

        meta = {
            "file_path": paragraph.file_path,
            "rel_path": paragraph.rel_path,
            "note_title": paragraph.note_title,
            "section_heading": paragraph.section_heading or "",
            "paragraph_index": paragraph.paragraph_index,
        }

        self._collection.add(
            ids=[doc_id],
            embeddings=[emb],
            metadatas=[meta],
            documents=[paragraph.text],
        )

