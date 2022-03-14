from agent import Agent
from environment import Env


class Curriculum:
    def __init__(self, tasks: list, tasks_length: list) -> None:
        self.tasks = tasks
        self.tasks_length = tasks_length


class AgentCuriculum:
    def __init__(self, agent: Agent, curriculum: Curriculum, env: Env) -> None:
        self.agent = agent
        self.curriculum = curriculum
        self.env = env

    def train_agent_on_task(self, task):
        for _ in range(self.curriculum.tasks_length[task]):
            self.agent.train_one_epoch(
                self.env,
            )
