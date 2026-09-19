from typing import NamedTuple
from hmz.flows import Agent, flow
from pydantic import BaseModel, Field


class Agents(NamedTuple):
    optimizer: Agent


class Config(BaseModel):
    rounds: int = Field(default=1, ge=1)


@flow()
def run(agents: Agents, task: str, config: Config | None = None):
    for _ in range((config or Config()).rounds):
        agents.optimizer(task)
