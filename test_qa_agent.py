from qa_agent import QAAgent


def main():

    print("=" * 80)
    print("GUTERMANN RAG QA AGENT")
    print("=" * 80)

    print("\nInitializing agent...")

    agent = QAAgent(
        top_k=4,
    )

    print("Agent ready.")

    # --------------------------------------------------------
    # Test 1 - Greeting
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST 1 - GREETING")
    print("=" * 80)

    message = "Hi"

    print(f"\nUSER: {message}")

    answer = agent.ask(message)

    print(f"AGENT: {answer}")

    # --------------------------------------------------------
    # Test 2 - Product question
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST 2 - QUESTION ANSWERING")
    print("=" * 80)

    message = (
        "Which products use True Sound Sensors?"
    )

    print(f"\nUSER: {message}")

    answer = agent.ask(message)

    print(f"AGENT: {answer}")

    # --------------------------------------------------------
    # Test 3 - Explicit follow-up
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST 3 - FOLLOW-UP")
    print("=" * 80)

    message = "What about the second one?"

    print(f"\nUSER: {message}")

    answer = agent.ask(message)

    print(f"AGENT: {answer}")

    # --------------------------------------------------------
    # Test 4 - Independent question
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST 4 - NEW QUESTION")
    print("=" * 80)

    message = (
        "What is the Minimum Level Profiling feature "
        "and which product has it?"
    )

    print(f"\nUSER: {message}")

    answer = agent.ask(message)

    print(f"AGENT: {answer}")

    # --------------------------------------------------------
    # Test 5 - Out of scope
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST 5 - OUT OF SCOPE")
    print("=" * 80)

    message = "What is the capital of France?"

    print(f"\nUSER: {message}")

    answer = agent.ask(message)

    print(f"AGENT: {answer}")

    # --------------------------------------------------------
    # Test 6 - Ambiguous follow-up
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST 6 - AMBIGUOUS FOLLOW-UP")
    print("=" * 80)

    message = "Tell me more"

    print(f"\nUSER: {message}")

    answer = agent.ask(message)

    print(f"AGENT: {answer}")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()