#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
from hmz.agents.claude import ClaudeCodeAgent, ClaudeCodeAgentConfig
from hmz.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1, help="attempts; 0 runs until stopped")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    agent = ClaudeCodeAgent(ClaudeCodeAgentConfig(model="claude-fable-5-1", effort="high", goals=False))
    runner = Runner(root / "agent/flow.py", [agent], {"rounds": args.rounds})
    if args.check:
        print("Valid: container Claude Code, claude-fable-5-1, high; no model request sent")
        return
    runner.run((root / "agent/TASK.md").read_text())


if __name__ == "__main__":
    main()
