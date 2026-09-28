from src.qa_agent import QAAgent


def main():
    print("=" * 80)
    print("GUTERMANN RAG QA AGENT")
    print("=" * 80)
    print()

    print("Initializing agent...")

    agent = QAAgent()

    print()
    print("=" * 80)

    # ============================================================
    # TEST 1 - GREETING
    # ============================================================

    print("TEST 1 - GREETING")
    print("=" * 80)

    query = "Hi"

    print()
    print(f"USER: {query}")

    answer = agent.answer(query)

    print(f"AGENT: {answer}")

    # ============================================================
    # TEST 2 - QUESTION ANSWERING
    # ============================================================

    print()
    print("=" * 80)
    print("TEST 2 - QUESTION ANSWERING")
    print("=" * 80)

    query = "Which products use True Sound Sensors?"

    print()
    print(f"USER: {query}")

    answer = agent.answer(query)

    print(f"AGENT: {answer}")

    # ============================================================
    # TEST 3 - FOLLOW-UP
    # ============================================================

    print()
    print("=" * 80)
    print("TEST 3 - FOLLOW-UP")
    print("=" * 80)

    query = "What about the second one?"

    print()
    print(f"USER: {query}")

    answer = agent.answer(query)

    print(f"AGENT: {answer}")

    # ============================================================
    # TEST 4 - NEW QUESTION
    # ============================================================

    print()
    print("=" * 80)
    print("TEST 4 - NEW QUESTION")
    print("=" * 80)

    query = (
        "What is the Minimum Level Profiling feature "
        "and which product has it?"
    )

    print()
    print(f"USER: {query}")

    answer = agent.answer(query)

    print(f"AGENT: {answer}")

    # ============================================================
    # TEST 5 - OUT OF SCOPE
    # ============================================================

    print()
    print("=" * 80)
    print("TEST 5 - OUT OF SCOPE")
    print("=" * 80)

    query = "What is the capital of France?"

    print()
    print(f"USER: {query}")

    answer = agent.answer(query)

    print(f"AGENT: {answer}")

    # ============================================================
    # TEST 6 - AMBIGUOUS FOLLOW-UP
    # ============================================================

    print()
    print("=" * 80)
    print("TEST 6 - AMBIGUOUS FOLLOW-UP")
    print("=" * 80)

    query = "Tell me more"

    print()
    print(f"USER: {query}")

    answer = agent.answer(query)

    print(f"AGENT: {answer}")

    # ============================================================
    # COMPLETE
    # ============================================================

    print()
    print("=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()