from qa_agent import QAAgent


def print_separator():
    print("\n" + "=" * 80)


def run_test(
    agent: QAAgent,
    number: int,
    question: str,
):
    print_separator()
    print(f"TEST {number}")
    print_separator()

    print("\nUSER:")
    print(question)

    print("\nAGENT:")

    answer = agent.ask(question)

    print(answer)


def main():
    print("=" * 80)
    print("GUTERMANN OFFICIAL PROBLEM STATEMENT TEST")
    print("=" * 80)

    print("\nInitializing agent...")

    try:
        agent = QAAgent()

    except Exception as error:
        print("\nFAILED TO INITIALIZE AGENT")

        print(
            f"{type(error).__name__}: {error}"
        )

        return

    print("Agent ready.")

    # ========================================================
    # TEST 1
    # Direct feature lookup
    # ========================================================

    agent.clear_memory()

    run_test(
        agent,
        1,
        "What is the Minimum Level Profiling feature and which product has it?",
    )

    # ========================================================
    # TEST 2
    # Product comparison
    # ========================================================

    agent.clear_memory()

    run_test(
        agent,
        2,
        "What is the difference between the AQUASCAN 610 and the AQUASCAN 760T?",
    )

    # ========================================================
    # TEST 3
    # Multi-product feature trace
    # ========================================================

    agent.clear_memory()

    run_test(
        agent,
        3,
        "Which products use True Sound Sensors (TSS) — and what advantage does TSS give?",
    )

    # ========================================================
    # TEST 4
    # Implicit recommendation
    # ========================================================

    agent.clear_memory()

    run_test(
        agent,
        4,
        "I need to find leaks on plastic pipes over long distances — what do you recommend?",
    )

    # ========================================================
    # TEST 5
    # Implicit recommendation
    # ========================================================

    agent.clear_memory()

    run_test(
        agent,
        5,
        "We want permanent monitoring in underground chambers with no drilling — what fits?",
    )

    # ========================================================
    # TEST 6
    # Hallucination probe
    # ========================================================

    agent.clear_memory()

    run_test(
        agent,
        6,
        "Can I order the ZONESCAN HYDRO today?",
    )

    # ========================================================
    # TEST 7
    # Hallucination probe
    # ========================================================

    agent.clear_memory()

    run_test(
        agent,
        7,
        "Does the ZONESCAN AI use hydrophone technology?",
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print_separator()

    print(
        "OFFICIAL PROBLEM STATEMENT TESTING COMPLETE"
    )

    print_separator()


if __name__ == "__main__":
    main()