#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
from hmz.agents.dsh import DshAgent, DshAgentConfig
from hmz.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1, help="attempts; 0 runs until stopped")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    agent = DshAgent(DshAgentConfig(model="deepseek-v4-flash", effort="max", goals=False))
    runner = Runner(root / "agent/flow.py", [agent], {"rounds": args.rounds})
    if args.check:
        print("Valid: DeepSeek Harness, deepseek-v4-flash, max; no model request sent")
        return
    runner.run((root / "agent/TASK.md").read_text())


if __name__ == "__main__":
    main()
