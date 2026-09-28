from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from vector_store import VectorStore


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
    def _tokenize(text: str) -> List[str]:
        """
        Basic tokenization used by BM25.
        """

        return text.lower().split()

    def _build_bm25_index(self):
        """
        Load all documents from ChromaDB and build the BM25 index.
        """

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
            self._tokenize(document.page_content)
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
        """
        Normalize scores to the 0-1 range using min-max scaling.
        """

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
            ) / (
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
        """
        Create a stable identifier for a product document.
        """

        product = document.metadata.get(
            "product",
            "",
        )

        category = document.metadata.get(
            "category",
            "",
        )

        return f"{category}::{product}"

    # ============================================================
    # VECTOR SEARCH
    # ============================================================

    def vector_search(
        self,
        query: str,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using dense vector similarity.
        """

        query_embedding = (
            self.vector_store.embeddings.embed_query(
                query
            )
        )

        results = self.vector_store.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=min(
                self.vector_k,
                len(self.documents),
            ),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        output = []

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            document = Document(
                page_content=text,
                metadata=metadata,
            )

            # Convert Chroma distance into a simple similarity score.
            similarity = 1.0 / (
                1.0 + float(distance)
            )

            output.append(
                (
                    document,
                    similarity,
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
        """
        Retrieve documents using BM25 lexical matching.
        """

        query_tokens = self._tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        output = []

        for index in ranked_indices[
            : self.bm25_k
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
        """
        Combine vector and BM25 retrieval.

        Vector and BM25 scores are independently normalized
        before applying their configured weights.

        Final score:

            hybrid_score =
                vector_weight * normalized_vector_score
                +
                bm25_weight * normalized_bm25_score
        """

        vector_results = self.vector_search(
            query
        )

        bm25_results = self.bm25_search(
            query
        )

        # --------------------------------------------------------
        # Store scores by stable document ID
        # --------------------------------------------------------

        vector_scores = {
            self._document_id(document): score
            for document, score in vector_results
        }

        bm25_scores = {
            self._document_id(document): score
            for document, score in bm25_results
        }

        # --------------------------------------------------------
        # Normalize individual retrieval scores
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Combine documents from both retrieval methods
        # --------------------------------------------------------

        documents_by_id: Dict[
            str,
            Document,
        ] = {}

        for document, _ in vector_results:
            documents_by_id[
                self._document_id(document)
            ] = document

        for document, _ in bm25_results:
            documents_by_id[
                self._document_id(document)
            ] = document

        # --------------------------------------------------------
        # Calculate hybrid scores
        # --------------------------------------------------------

        combined_scores: Dict[
            str,
            float,
        ] = {}

        for document_id in documents_by_id:

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

        # --------------------------------------------------------
        # Rank by combined score
        # --------------------------------------------------------

        ranked = sorted(
            combined_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        # --------------------------------------------------------
        # Return controlled TOP_K results
        # --------------------------------------------------------

        return [
            (
                documents_by_id[
                    document_id
                ],
                score,
            )
            for document_id, score in ranked[
                :top_k
            ]
        ]


# ============================================================
# Standalone Retrieval Test
# ============================================================

if __name__ == "__main__":

    store = VectorStore()

    retriever = HybridRetriever(
        vector_store=store,
    )

    query = (
        "Which products use True Sound Sensors?"
    )

    results = retriever.hybrid_search(
        query=query,
        top_k=4,
    )

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 80)

    print(f"Query: {query}")

    for rank, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 80)

        print(
            f"Rank     : {rank}"
        )

        print(
            f"Score    : {score:.4f}"
        )

        print(
            f"Product  : "
            f"{document.metadata.get('product')}"
        )

        print(
            f"Category : "
            f"{document.metadata.get('category')}"
        )