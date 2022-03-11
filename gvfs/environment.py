import numpy as np
from gym.spaces import Box, Discrete
from mazelab import BaseEnv, BaseMaze
from mazelab import DeepMindColor as color
from mazelab import Object, VonNeumannMotion

# The code depends on the "maze_config" variable, which is external to the classes in the script
# This was done to ease the interaction with the gym module
# 'maze_config' is changed in main.py when the maze is updated
maze_config = {"shape": np.array([0, 0]), "start_idx": [0], "goal_idx": [0]}


class Maze(BaseMaze):
    @property
    def size(self):
        return maze_config["shape"].shape

    def make_objects(self):
        free = Object(
            "free",
            0,
            color.free,
            False,
            np.stack(np.where(maze_config["shape"] == 0), axis=1),
        )
        obstacle = Object(
            "obstacle",
            1,
            color.obstacle,
            True,
            np.stack(np.where(maze_config["shape"] == 1), axis=1),
        )
        agent = Object("agent", 2, color.agent, False, [])
        goal = Object("goal", 3, color.goal, False, [])
        return free, obstacle, agent, goal


class Env(BaseEnv):
    def __init__(self):
        super().__init__()

        self.maze = Maze()
        self.motions = VonNeumannMotion()

        self.observation_space = Box(
            low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8
        )
        self.action_space = Discrete(len(self.motions))

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [
            current_position[0] + motion[0],
            current_position[1] + motion[1],
        ]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +20
            done = True
        elif not valid:
            reward = -2
            done = False
        else:
            reward = -1
            done = False
        return self.maze.to_value(), reward, done, {}

    def reset(self):
        self.maze.objects.agent.positions = maze_config["start_idx"]
        self.maze.objects.goal.positions = maze_config["goal_idx"]
        return self.maze.to_value()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = (
            position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        )
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()
