from triage import TriageService


def main():

    service = TriageService()

    tests = [
        {
            "message": "Hi",
            "history": False,
        },
        {
            "message": "Hello there!",
            "history": False,
        },
        {
            "message": "Which products use True Sound Sensors?",
            "history": False,
        },
        {
            "message": "What is the Minimum Level Profiling feature?",
            "history": False,
        },
        {
            "message": "What about the second one?",
            "history": True,
        },
        {
            "message": "Tell me more",
            "history": True,
        },
        {
            "message": "What about that?",
            "history": True,
        },
        {
            "message": "What is the capital of France?",
            "history": False,
        },
        {
            "message": "Can you recommend a restaurant?",
            "history": False,
        },
    ]

    print("=" * 80)
    print("TRIAGE TEST")
    print("=" * 80)

    for test in tests:

        result = service.classify(
            message=test["message"],
            has_conversation_history=test["history"],
        )

        print("\n" + "-" * 80)

        print(
            f"Message    : {test['message']}"
        )

        print(
            f"History    : {test['history']}"
        )

        print(
            f"Category   : {result.category.value}"
        )

        print(
            f"Confidence : {result.confidence:.2f}"
        )

        print(
            f"Reason     : {result.reason}"
        )


if __name__ == "__main__":
    main()