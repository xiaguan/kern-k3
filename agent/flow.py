import itertools
import time
import traceback
from typing import NamedTuple
from hmz.flows import Agent, flow
from pydantic import BaseModel, Field


class Agents(NamedTuple):
    optimizer: Agent


class Config(BaseModel):
    rounds: int = Field(default=1, ge=0)


@flow()
def run(agents: Agents, task: str, config: Config | None = None):
    rounds = (config or Config()).rounds
    for index in itertools.count(1):
        if rounds and index > rounds:
            return
        print(f"Optimization round {index}", flush=True)
        try:
            agents.optimizer(task)
        except Exception:
            if rounds:
                raise
            traceback.print_exc()
            print("Round failed; retrying in 30 seconds", flush=True)
            time.sleep(30)
