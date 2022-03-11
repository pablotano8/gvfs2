# pylint: disable=missing-function-docstring
from abc import ABC, abstractmethod

import numpy as np


class Tabular(ABC):
    """Tabular objects should have a Reset function and a TD backup"""

    @abstractmethod
    def reset(self):
        """Reset values to zero"""

    @abstractmethod
    def td_backup(self, state, action, state_next, utility):
        """Apply single TD backup"""


class GVFTable(Tabular):
    """Table of hierarchical GVFs"""

    def __init__(
        self,
        shape,
        num_levels,
        num_actions,
        gamma=0.99,
        lr=0.1,
        num_primitives=4,
        primitives=None,
        values=None,
    ):
        self.shape = shape
        self.num_levels = num_levels
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = lr
        self.primitives = primitives
        self.num_primitives = num_primitives
        self.values = (
            values
            if values is not None
            else np.zeros(
                (self.shape[0], self.shape[1], self.num_levels, self.num_primitives)
            )
        )

    def reset(self):
        self.values = np.zeros(
            (self.shape[0], self.shape[1], self.num_levels, self.num_primitives)
        )

    def td_backup(self, state, action, state_next, utility):
        for prim in range(self.num_primitives):
            importance_weight = self.primitives[prim, action]
            self.values[state[0], state[1], 0, prim] = self.values[
                state[0], state[1], 0, prim
            ] + (
                self.lr
                * importance_weight
                * (
                    utility
                    + self.gamma * self.values[state_next[0], state_next[1], 0, prim]
                    - self.values[state[0], state[1], 0, prim]
                )
            )

        for level in range(1, self.num_levels):
            for prim in range(self.num_primitives):
                importance_weight = self.primitives[prim, action]
                utility = np.max(
                    self.values[state_next[0], state_next[1], level - 1, :]
                )
                self.values[state[0], state[1], level, prim] = self.values[
                    state[0], state[1], level, prim
                ] + (
                    self.lr
                    * importance_weight
                    * (
                        utility
                        + self.gamma
                        * self.values[state_next[0], state_next[1], level, prim]
                        - self.values[state[0], state[1], level, prim]
                    )
                )

        self.values = np.clip(
            self.values, 0, 1
        )  # clip all values between 0 and 1 to avoid convergence issues


class QTable(Tabular):
    """Table of Q-values"""

    def __init__(self, shape, num_actions, gamma, lr, values=None):
        self.shape = shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = lr
        self.values = (
            values
            if values is not None
            else np.zeros((shape[0], shape[1], num_actions))
        )

    def reset(self):
        self.values = np.zeros((self.shape[0], self.shape[1], self.num_actions))

    def td_backup(self, state, action, state_next, utility):
        self.values[state[0], state[1], action] += self.lr * (
            utility
            + self.gamma * np.max(self.values[state_next[0], state_next[1], :])
            - self.values[state[0], state[1], action]
        )
