from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConversationTurn:
    """Stores one complete user/assistant interaction."""

    user_message: str
    assistant_answer: str

    retrieved_products: List[str] = field(
        default_factory=list
    )

    mentioned_products: List[str] = field(
        default_factory=list
    )


class ConversationMemory:
    """
    Lightweight in-memory conversation history.

    Stores the most recent conversation turns and separates
    retrieved products from products actually discussed in
    the assistant's answer.
    """

    def __init__(self, max_turns: int = 10):
        if max_turns < 1:
            raise ValueError(
                "max_turns must be at least 1."
            )

        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []

    def add_turn(
        self,
        user_message: str,
        assistant_answer: str,
        retrieved_products: Optional[List[str]] = None,
        mentioned_products: Optional[List[str]] = None,
    ) -> None:
        """Store a completed conversation turn."""

        turn = ConversationTurn(
            user_message=user_message,
            assistant_answer=assistant_answer,
            retrieved_products=retrieved_products or [],
            mentioned_products=mentioned_products or [],
        )

        self.turns.append(turn)

        # Keep only the most recent turns.
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def has_history(self) -> bool:
        """Return True if conversation history exists."""

        return bool(self.turns)

    def get_last_turn(
        self,
    ) -> Optional[ConversationTurn]:
        """Return the latest conversation turn."""

        if not self.turns:
            return None

        return self.turns[-1]

    def get_last_products(self) -> List[str]:
        """Return products retrieved during the latest turn."""

        last_turn = self.get_last_turn()

        if last_turn is None:
            return []

        return last_turn.retrieved_products

    def get_last_mentioned_products(self) -> List[str]:
        """Return products mentioned in the latest answer."""

        last_turn = self.get_last_turn()

        if last_turn is None:
            return []

        return last_turn.mentioned_products

    def get_latest_product_context(
        self,
    ) -> Optional[ConversationTurn]:
        """
        Search backwards for the most recent turn containing
        useful product context.

        Mentioned products are preferred because they represent
        what was actually discussed in the answer.
        """

        for turn in reversed(self.turns):

            if turn.mentioned_products:
                return turn

            if turn.retrieved_products:
                return turn

        return None

    def get_recent_turns(
        self,
        count: int = 3,
    ) -> List[ConversationTurn]:
        """Return the most recent N conversation turns."""

        if count < 1:
            return []

        return self.turns[-count:]

    def get_context(
        self,
        count: Optional[int] = None,
    ) -> str:
        """Return conversation history as readable text."""

        if not self.turns:
            return "No previous conversation."

        turns = self.turns

        if count is not None:

            if count < 1:
                return "No previous conversation."

            turns = self.turns[-count:]

        lines = []

        for index, turn in enumerate(
            turns,
            start=1,
        ):
            retrieved = (
                ", ".join(turn.retrieved_products)
                if turn.retrieved_products
                else "None"
            )

            mentioned = (
                ", ".join(turn.mentioned_products)
                if turn.mentioned_products
                else "None"
            )

            lines.append(
                f"Turn {index}\n"
                f"User: {turn.user_message}\n"
                f"Assistant: {turn.assistant_answer}\n"
                f"Retrieved products: {retrieved}\n"
                f"Mentioned products: {mentioned}"
            )

        return "\n\n".join(lines)

    def clear(self) -> None:
        """Clear all conversation history."""

        self.turns.clear()