#!/usr/bin/env python3
"""Simple command-line interface for querying the BasicAgent.

Usage
-----
$ python query_agent.py "Your question here"
# or, without an argument, it will prompt interactively.
"""
from __future__ import annotations

import sys
import os
from app import BasicAgent

def main() -> None:
    # Ensure your environment variable is set:  export OPENAI_API_KEY=...
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("👤  Question: ")
    agent = BasicAgent()           # customise provider/model as needed
    answer = agent(question)
    print(f"🤖  Answer: {answer}")

if __name__ == "__main__":
    main()