from src.memory import ConversationMemory
from src.follow_up import FollowUpResolver


def print_result(
    original: str,
    result,
):
    print("\n" + "-" * 80)

    print(
        f"Original : {original}"
    )

    print(
        f"Resolved : {result.resolved_query}"
    )

    print(
        f"Clarify  : {result.needs_clarification}"
    )

    if result.needs_clarification:

        print(
            f"Question : "
            f"{result.clarification_question}"
        )


def main():

    memory = ConversationMemory()

    resolver = FollowUpResolver()

    # --------------------------------------------------------
    # First conversation turn
    # --------------------------------------------------------

    first_question = (
        "Which products use True Sound Sensors?"
    )

    first_answer = (
        "The AQUASCAN 760T and AQUASCAN TM3 "
        "use True Sound Sensors."
    )

    retrieved_products = [
        "AQUASCAN 760T",
        "AQUASCAN TM3",
        "AQUASCAN 620L",
        "AQUASCOPE 3",
    ]

    memory.add_turn(
        user_message=first_question,
        assistant_answer=first_answer,
        retrieved_products=retrieved_products,
    )

    print("=" * 80)
    print("CONVERSATION MEMORY")
    print("=" * 80)

    print(
        memory.get_context()
    )

    # --------------------------------------------------------
    # Test 1 - Explicit second product
    # --------------------------------------------------------

    follow_up = "What about the second one?"

    print("\n" + "=" * 80)
    print("FOLLOW-UP TEST 1")
    print("=" * 80)

    result = resolver.resolve(
        follow_up,
        memory,
    )

    print_result(
        follow_up,
        result,
    )

    # --------------------------------------------------------
    # Test 2 - Ambiguous "Tell me more"
    # --------------------------------------------------------

    follow_up = "Tell me more"

    print("\n" + "=" * 80)
    print("FOLLOW-UP TEST 2")
    print("=" * 80)

    result = resolver.resolve(
        follow_up,
        memory,
    )

    print_result(
        follow_up,
        result,
    )

    # --------------------------------------------------------
    # Test 3 - Explicit first product
    # --------------------------------------------------------

    follow_up = "What about the first product?"

    print("\n" + "=" * 80)
    print("FOLLOW-UP TEST 3")
    print("=" * 80)

    result = resolver.resolve(
        follow_up,
        memory,
    )

    print_result(
        follow_up,
        result,
    )

    # --------------------------------------------------------
    # Test 4 - Ambiguous generic reference
    # --------------------------------------------------------

    follow_up = "Tell me more about that product."

    print("\n" + "=" * 80)
    print("FOLLOW-UP TEST 4")
    print("=" * 80)

    result = resolver.resolve(
        follow_up,
        memory,
    )

    print_result(
        follow_up,
        result,
    )


if __name__ == "__main__":
    main()