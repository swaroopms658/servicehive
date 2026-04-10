from __future__ import annotations

import os

from dotenv import load_dotenv

from app.graph import build_graph, invoke_turn


def main() -> int:
    load_dotenv()

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    if not has_openai and not has_groq:
        print("No API key found.")
        print("Add OPENAI_API_KEY or GROQ_API_KEY to the .env file before running the agent.")
        return 1

    graph = build_graph()
    thread_id = "autostream-cli-demo"

    print("AutoStream assistant")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        state = invoke_turn(graph, thread_id=thread_id, user_message=user_input)
        print(f"Agent: {state['reply']}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
