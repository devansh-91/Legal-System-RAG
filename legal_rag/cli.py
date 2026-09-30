"""Chat with the legal assistant from the terminal:  python -m legal_rag.cli"""

from __future__ import annotations

import sys

from .chain import LegalRAG
from .prompts import DISCLAIMER


def main() -> None:
    rag = LegalRAG.from_settings()
    history: list[dict[str, str]] = []
    print("Nyaya Sahayak — Indian legal help. Type 'exit' to quit.\n" + DISCLAIMER + "\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit", "q"}:
            return
        if not question:
            continue
        answer = []
        sources = []
        print("\nAssistant: ", end="", flush=True)
        for event in rag.stream(question, history):
            if event["type"] == "meta":
                sources = event["sources"]
                for notice in event["notices"]:
                    print(f"\n⚠️  {notice}\n", flush=True)
            elif event["type"] == "token":
                answer.append(event["content"])
                sys.stdout.write(event["content"])
                sys.stdout.flush()
        print("\n\nSources:")
        for s in sources:
            section = f" › {s['section']}" if s["section"] else ""
            print(f"  [{s['id']}] {s['title']}{section}  ({s['source']})")
        print()
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "".join(answer)},
        ]


if __name__ == "__main__":
    main()
