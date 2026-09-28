from src.vector_store import VectorStore
from src.retrieval import HybridRetriever


def print_results(
    title,
    results,
):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for rank, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n--- Rank {rank} ---"
        )

        print(
            f"Score   : {score:.4f}"
        )

        print(
            f"Product : "
            f"{document.metadata.get('product')}"
        )

        print(
            f"Category: "
            f"{document.metadata.get('category')}"
        )


def main():

    store = VectorStore()

    retriever = HybridRetriever(
        vector_store=store,
    )

    queries = [
        "What is the Minimum Level Profiling feature and which product has it?",
        "Which products use True Sound Sensors?",
        "I need to find leaks on plastic pipes over long distances. What do you recommend?",
    ]

    for query in queries:

        print("\n\n" + "#" * 80)
        print(
            f"QUERY: {query}"
        )
        print("#" * 80)

        # ----------------------------------------------------
        # Vector retrieval
        # ----------------------------------------------------

        vector_results = retriever.vector_search(
            query
        )

        print_results(
            "VECTOR RETRIEVAL",
            vector_results,
        )

        # ----------------------------------------------------
        # BM25 retrieval
        # ----------------------------------------------------

        bm25_results = retriever.bm25_search(
            query
        )

        print_results(
            "BM25 RETRIEVAL",
            bm25_results,
        )

        # ----------------------------------------------------
        # Hybrid retrieval
        # ----------------------------------------------------

        hybrid_results = retriever.hybrid_search(
            query=query,
            top_k=4,
        )

        print_results(
            "HYBRID RETRIEVAL",
            hybrid_results,
        )


if __name__ == "__main__":
    main()