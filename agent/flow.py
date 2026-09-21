"""Four optimizers in parallel, one per area, each in its own worktree and branch.

Every agent keeps one conversation across rounds (a new one only after a
failed turn), writes its own session log, and shares the GPUs through the
lock the eval commands take. Nothing here merges: the branches are composed
onto main by `compose.py` when somebody decides to.
"""

import asyncio
import itertools
import traceback
from pathlib import Path
from typing import NamedTuple

from hmz.flows import Agent, flow
from pydantic import BaseModel, Field


class Agents(NamedTuple):
    a: Agent
    b: Agent
    c: Agent
    d: Agent


class Config(BaseModel):
    rounds: int = Field(default=0, ge=0)
    worktrees: str = "/home/worker/wt"
    records: str = "/home/worker/bench_results"


def opening(task: str, name: str, area: str, records: str) -> str:
    return (
        f"{task}\n\n## 你的分区（agent {name}）\n\n{area}\n\n"
        f"你的记录目录是 {records}/{name}/，STATE.md 也在那里。"
        f"你的工作树是当前目录，分支 agent/{name}。这个对话会跨轮保留，下一轮只会收到一句开始的提示。\n"
    )


def consume(session, prompt: str, log: Path) -> str:
    with log.open("a", encoding="utf-8") as sink:
        sink.write(f"\n\n===== turn =====\n{prompt[:200]}\n-----\n")
        last = ""
        for event in session.stream(prompt):
            kind = event.kind
            if kind in ("text", "reasoning"):
                sink.write(event.text)
            elif kind == "tool":
                sink.write(f"\n>> {event.text[:400]}\n")
            elif kind in ("result", "failed"):
                last = event.text
                sink.write(f"\n===== {kind} =====\n{event.text}\n")
            sink.flush()
        return last


async def drive(name: str, agent: Agent, task: str, config: Config) -> None:
    cwd = Path(config.worktrees, name)
    area = (cwd / "agent/areas" / f"{name}.md").read_text(encoding="utf-8")
    records = Path(config.records, name)
    records.mkdir(parents=True, exist_ok=True)
    log = records / "session.log"
    session = agent.new(cwd)
    fresh = True
    for index in itertools.count(1):
        if config.rounds and index > config.rounds:
            return
        print(f"[{name}] round {index}", flush=True)
        prompt = (
            opening(task, name, area, config.records)
            if fresh
            else f"开始第 {index} 轮。先看 STATE.md 和 git status，再按任务书走。"
        )
        try:
            await asyncio.to_thread(consume, session, prompt, log)
            fresh = False
        except Exception:
            traceback.print_exc()
            if config.rounds:
                raise
            print(f"[{name}] round {index} failed; new session in 30 s", flush=True)
            await asyncio.sleep(30)
            session = agent.new(cwd)
            fresh = True


@flow()
async def run(agents: Agents, task: str, config: Config | None = None):
    cfg = config or Config()
    await asyncio.gather(
        *(drive(name, agent, task, cfg) for name, agent in zip(Agents._fields, agents))
    )
