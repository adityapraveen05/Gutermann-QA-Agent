from src.qa_agent import QAAgent


def main():
    print("=" * 80)
    print("GUTERMANN OFFICIAL PROBLEM STATEMENT TEST")
    print("=" * 80)
    print()

    print("Initializing agent...")

    agent = QAAgent()

    print("Agent ready.")

    # ============================================================
    # TEST 1
    # ============================================================

    print()
    print("=" * 80)
    print("TEST 1")
    print("=" * 80)

    query = (
        "What is the Minimum Level Profiling feature "
        "and which product has it?"
    )

    print()
    print("USER:")
    print(query)

    answer = agent.answer(query)

    print()
    print("AGENT:")
    print(answer)

    # ============================================================
    # TEST 2
    # ============================================================

    agent.memory.clear()

    print()
    print("=" * 80)
    print("TEST 2")
    print("=" * 80)

    query = (
        "What is the difference between the "
        "AQUASCAN 610 and the AQUASCAN 760T?"
    )

    print()
    print("USER:")
    print(query)

    answer = agent.answer(query)

    print()
    print("AGENT:")
    print(answer)

    # ============================================================
    # TEST 3
    # ============================================================

    agent.memory.clear()

    print()
    print("=" * 80)
    print("TEST 3")
    print("=" * 80)

    query = (
        "Which products use True Sound Sensors (TSS) "
        "— and what advantage does TSS give?"
    )

    print()
    print("USER:")
    print(query)

    answer = agent.answer(query)

    print()
    print("AGENT:")
    print(answer)

    # ============================================================
    # TEST 4
    # ============================================================

    agent.memory.clear()

    print()
    print("=" * 80)
    print("TEST 4")
    print("=" * 80)

    query = (
        "I need to find leaks on plastic pipes "
        "over long distances — what do you recommend?"
    )

    print()
    print("USER:")
    print(query)

    answer = agent.answer(query)

    print()
    print("AGENT:")
    print(answer)

    # ============================================================
    # TEST 5
    # ============================================================

    agent.memory.clear()

    print()
    print("=" * 80)
    print("TEST 5")
    print("=" * 80)

    query = (
        "We want permanent monitoring in underground "
        "chambers with no drilling — what fits?"
    )

    print()
    print("USER:")
    print(query)

    answer = agent.answer(query)

    print()
    print("AGENT:")
    print(answer)

    # ============================================================
    # TEST 6
    # ============================================================

    agent.memory.clear()

    print()
    print("=" * 80)
    print("TEST 6")
    print("=" * 80)

    query = "Can I order the ZONESCAN HYDRO today?"

    print()
    print("USER:")
    print(query)

    answer = agent.answer(query)

    print()
    print("AGENT:")
    print(answer)

    # ============================================================
    # TEST 7
    # ============================================================

    agent.memory.clear()

    print()
    print("=" * 80)
    print("TEST 7")
    print("=" * 80)

    query = "Does the ZONESCAN AI use hydrophone technology?"

    print()
    print("USER:")
    print(query)

    answer = agent.answer(query)

    print()
    print("AGENT:")
    print(answer)

    # ============================================================
    # COMPLETE
    # ============================================================

    print()
    print("=" * 80)
    print("OFFICIAL PROBLEM STATEMENT TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()