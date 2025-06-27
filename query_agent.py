#!/usr/bin/env python3
"""Persistent command-line interface for querying the ``BasicAgent``.

Usage examples
--------------
1. Single-shot question (script exits after answering):

   $ python query_agent.py "Who is the president of the US?"

2. Interactive REPL (ask as many questions as you like):

   $ python query_agent.py

   👤  Question:  Who wrote *The Hobbit*?
   🤖  Answer:    J. R. R. Tolkien

   👤  Question:  exit  # or Ctrl-D to quit
   👋  Goodbye!
"""
from __future__ import annotations

import sys
import os
from app import BasicAgent
import readline  # Adds history/navigation support on *nix systems
from typing import Set
import itertools
import threading
import time

EXIT_COMMANDS: Set[str] = {"exit", "quit", "q", ":exit", ":quit", ":q"}


def _ensure_openai_api_key() -> None:
    """Validate that the required OPENAI_API_KEY environment variable is set.

    Raises:
        EnvironmentError: If the environment variable is missing.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Run `export OPENAI_API_KEY=your_api_key` and try again."
        )


def _interactive_loop(agent: "BasicAgent") -> None:
    """Run a persistent REPL for querying *agent* until the user exits.

    Args:
        agent: An instantiated BasicAgent ready to answer questions.
    """
    print("\n📝  Enter your questions below. Type 'exit' or press Ctrl-D to quit.\n")

    while True:
        try:
            question: str = input("👤  Question: ").strip()
        except EOFError:  # Handles Ctrl-D gracefully
            print("\n👋  Goodbye!")
            break
        except KeyboardInterrupt:  # Handles Ctrl-C gracefully
            print("\nInterrupted by user. Exiting…")
            break

        if not question:
            # Ignore empty inputs so the loop continues.
            continue

        if question.lower() in EXIT_COMMANDS:
            print("👋  Goodbye!")
            break

        with _Spinner("Processing"):
            answer: str = agent(question)
        print(f"🤖  Answer: {answer}\n")


class _Spinner:
    """Simple CLI spinner displayed while awaiting the agent's response.

    Usage:
        with _Spinner("Processing"):
            answer = agent(question)
    """

    def __init__(self, message: str = "Processing") -> None:
        self._message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:  # pragma: no cover — visual effect
        for frame in itertools.cycle("|/-\\"):
            if self._stop_event.is_set():
                break
            sys.stdout.write(f"\r{self._message}... {frame}")
            sys.stdout.flush()
            time.sleep(0.1)
        # Clear the line after stopping
        sys.stdout.write("\r" + " " * (len(self._message) + 5) + "\r")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join()
        # Returning False so that any exception propagates
        return False


def main() -> None:
    _ensure_openai_api_key()

    # Instantiate the agent **once** so it can be reused across questions.
    agent = BasicAgent()  # customise provider/model as needed

    # If the user passed a question as CLI arguments, answer it and exit.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        with _Spinner("Processing"):
            answer = agent(question)
        print(f"🤖  Answer: {answer}")
        return

    # Otherwise, fall back to an interactive persistent loop.
    _interactive_loop(agent)


if __name__ == "__main__":
    main()