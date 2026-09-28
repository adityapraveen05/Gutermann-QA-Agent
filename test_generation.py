from vector_store import VectorStore
from retrieval import HybridRetriever
from generation import generate_answer, OllamaGenerationError


def main():

    # --------------------------------------------------------
    # Vector store
    # --------------------------------------------------------

    store = VectorStore()

    # Make sure the current knowledge base is indexed.
    store.rebuild()

    # --------------------------------------------------------
    # Hybrid retriever
    # --------------------------------------------------------

    retriever = HybridRetriever(
        vector_store=store,
        vector_k=10,
        bm25_k=10,
        vector_weight=0.5,
        bm25_weight=0.5,
    )

    # --------------------------------------------------------
    # Test questions
    # --------------------------------------------------------

    queries = [
        "Which products use True Sound Sensors?",
        (
            "What is the Minimum Level Profiling feature "
            "and which product has it?"
        ),
        (
            "I need to find leaks on plastic pipes over "
            "long distances. What do you recommend?"
        ),
    ]

    for query in queries:

        print("\n" + "=" * 80)
        print(f"QUESTION: {query}")
        print("=" * 80)

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        results = retriever.hybrid_search(
            query=query,
            top_k=4,
        )

        documents = [
            document
            for document, score in results
        ]

        print("\nRetrieved products:")

        for index, document in enumerate(
            documents,
            start=1,
        ):
            print(
                f"{index}. "
                f"{document.metadata.get('product')}"
            )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        try:

            answer = generate_answer(
                query=query,
                documents=documents,
            )

            print("\nANSWER")
            print("-" * 80)
            print(answer)

        except OllamaGenerationError as error:

            print("\nGENERATION ERROR")
            print("-" * 80)
            print(error)


if __name__ == "__main__":
    main()