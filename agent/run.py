#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
from hmz.agents.codex import CodexAgent, CodexAgentConfig
from hmz.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    agent = CodexAgent(CodexAgentConfig(model="gpt-6-astra", effort="medium", goals=False))
    runner = Runner(root / "agent/flow.py", [agent], {"rounds": args.rounds})
    if args.check:
        print("Valid: container Codex, gpt-6-astra, medium; no model request sent")
        return
    runner.run((root / "agent/TASK.md").read_text())


if __name__ == "__main__":
    main()
