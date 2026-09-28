from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from .vector_store import VectorStore


class HybridRetriever:
    """
    Hybrid retriever combining:

    1. Dense semantic vector retrieval
    2. BM25 lexical retrieval

    The two retrieval scores are normalized and combined
    using configurable weights.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        vector_k: int = 10,
        bm25_k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):

        self.vector_store = vector_store

        self.vector_k = vector_k
        self.bm25_k = bm25_k

        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        self.documents: List[Document] = []
        self.bm25 = None

        self._build_bm25_index()

    # ============================================================
    # BM25 INDEX
    # ============================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> List[str]:
        """Basic tokenization used by BM25."""

        return text.lower().split()

    def _build_bm25_index(self):

        stored = self.vector_store.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        documents = stored.get(
            "documents",
            [],
        )

        metadatas = stored.get(
            "metadatas",
            [],
        )

        self.documents = [
            Document(
                page_content=text,
                metadata=metadata,
            )
            for text, metadata in zip(
                documents,
                metadatas,
            )
        ]

        if not self.documents:

            raise ValueError(
                "No documents found in ChromaDB. "
                "Run vector_store.rebuild() first."
            )

        tokenized_documents = [
            self._tokenize(
                document.page_content
            )
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    # ============================================================
    # SCORE NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_scores(
        scores: Dict[str, float],
    ) -> Dict[str, float]:

        if not scores:
            return {}

        values = list(
            scores.values()
        )

        minimum = min(values)
        maximum = max(values)

        if maximum == minimum:

            return {
                key: 1.0
                for key in scores
            }

        return {
            key: (
                value - minimum
            )
            / (
                maximum - minimum
            )
            for key, value in scores.items()
        }

    # ============================================================
    # DOCUMENT ID
    # ============================================================

    @staticmethod
    def _document_id(
        document: Document,
    ) -> str:

        product = document.metadata.get(
            "product",
            "",
        )

        category = document.metadata.get(
            "category",
            "",
        )

        return (
            f"{category}::{product}"
        )

    # ============================================================
    # VECTOR SEARCH
    # ============================================================

    def vector_search(
        self,
        query: str,
    ) -> List[Tuple[Document, float]]:

        query_embedding = (
            self.vector_store.embeddings.embed_query(
                query
            )
        )

        results = self.vector_store.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=self.vector_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = (
            results.get(
                "documents",
                [[]],
            )[0]
        )

        metadatas = (
            results.get(
                "metadatas",
                [[]],
            )[0]
        )

        distances = (
            results.get(
                "distances",
                [[]],
            )[0]
        )

        output = []

        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            document = Document(
                page_content=text,
                metadata=metadata,
            )

            score = 1.0 / (
                1.0 + distance
            )

            output.append(
                (
                    document,
                    score,
                )
            )

        return output

    # ============================================================
    # BM25 SEARCH
    # ============================================================

    def bm25_search(
        self,
        query: str,
    ) -> List[Tuple[Document, float]]:

        if not self.bm25:
            return []

        tokenized_query = self._tokenize(
            query
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        output = []

        for index in ranked_indices[
            :self.bm25_k
        ]:

            output.append(
                (
                    self.documents[index],
                    float(scores[index]),
                )
            )

        return output

    # ============================================================
    # HYBRID SEARCH
    # ============================================================

    def hybrid_search(
        self,
        query: str,
        top_k: int = 4,
    ) -> List[Tuple[Document, float]]:

        vector_results = (
            self.vector_search(query)
        )

        bm25_results = (
            self.bm25_search(query)
        )

        vector_scores = {
            self._document_id(document): score
            for document, score
            in vector_results
        }

        bm25_scores = {
            self._document_id(document): score
            for document, score
            in bm25_results
        }

        normalized_vector = (
            self._normalize_scores(
                vector_scores
            )
        )

        normalized_bm25 = (
            self._normalize_scores(
                bm25_scores
            )
        )

        documents_by_id = {}

        for document, _ in vector_results:

            documents_by_id[
                self._document_id(document)
            ] = document

        for document, _ in bm25_results:

            documents_by_id[
                self._document_id(document)
            ] = document

        combined_scores = {}

        all_ids = set(
            normalized_vector
        ) | set(
            normalized_bm25
        )

        for document_id in all_ids:

            vector_score = (
                normalized_vector.get(
                    document_id,
                    0.0,
                )
            )

            bm25_score = (
                normalized_bm25.get(
                    document_id,
                    0.0,
                )
            )

            combined_scores[
                document_id
            ] = (
                self.vector_weight
                * vector_score
                +
                self.bm25_weight
                * bm25_score
            )

        ranked = sorted(
            combined_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            (
                documents_by_id[document_id],
                score,
            )
            for document_id, score
            in ranked[:top_k]
        ]