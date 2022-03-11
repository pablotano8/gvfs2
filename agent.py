import numpy as np
import torch
from torch.distributions.categorical import Categorical

from actor import Actor
from environment import maze_config
from tabular import GVFTable


class Agent:
    """Acts in the environment and trains the gvfs and the actor"""

    def __init__(self, env, epsilon_explo=0.05, state=None, exploration=False) -> None:
        self.state = state
        self.epsilon_explo = epsilon_explo
        self.exploration = exploration
        self.num_actions = len(env.motions)
        self.action = 0
        self.beh = 0

    def controller(self, gvfs: GVFTable, threshold=0.05):
        """The controller chooses the action based on the gvfs"""
        if self.exploration is True or np.random.uniform(0, 1) < self.epsilon_explo:
            action = np.random.choice(range(self.num_actions))
        else:
            for level in range(gvfs.num_levels):
                if np.any(
                    gvfs.values[self.state[0], self.state[1], level, :] > threshold
                ):
                    self.beh = np.argmax(
                        gvfs.values[self.state[0], self.state[1], level, :]
                    )
                    break
                else:
                    self.beh = np.random.choice(range(len(gvfs.primitives)))

            action = np.random.choice(
                range(self.num_actions), p=gvfs.primitives[self.beh, :]
            )
        return action

    def train_one_epoch(
        self,
        env,
        gvfs: GVFTable,
        actor: Actor,
        actor_optimizer=False,
        batch_size=200,
    ):
        """Performs one epoch in the environment, updates the gvfs and the actor network"""
        batch_acts = []  # for actions
        batch_behs = []  # for behaviors
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns

        _, done, ep_rews = env.reset(), False, []  # initialiize episode variables

        while True:

            # The agent does a step in the environment:
            self.state = env.maze.objects.agent.positions[0] * 1  # position
            r_pos = env.maze.objects.goal.positions[0] * 1  # goal
            self.action = self.controller(gvfs)  # action
            _, rew, done, _ = env.step(self.action)  # one step in the environment
            pos_next = env.maze.objects.agent.positions[0] * 1  # next position

            ###### The primitives are required to update the gvfs, the prims. come from the actor net #############
            output_actor = torch.reshape(
                actor(torch.FloatTensor([1]).to("cpu")), (4, 4)
            )

            primitives = np.zeros((4, 4))
            for prim in range(gvfs.num_primitives):
                primitives[prim, :] = (
                    Categorical(logits=output_actor[prim])
                    .probs.detach()
                    .to("cpu")
                    .numpy()
                )
                primitives[prim, :] = primitives[prim, :] / np.sum(primitives[prim, :])

            gvfs.primitives = primitives
            ############################################

            # Update the GVFs
            utility = (pos_next[0] == r_pos[0] and pos_next[1] == r_pos[1]) * 1
            if self.exploration is True:
                gvfs.td_backup(self.state, self.action, pos_next, utility)

            # Save action, behavior and reward
            batch_acts.append(self.action)
            batch_behs.append(self.beh)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret = sum(ep_rews)
                batch_rets.append(ep_ret)

                # calculate cumulative reward from t to T ("rew to go")
                rtgs = np.zeros_like(ep_rews)
                for idx in reversed(range(len(ep_rews))):
                    rtgs[idx] = ep_rews[idx] + (
                        rtgs[idx + 1] if idx + 1 < len(ep_rews) else 0
                    )

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += list(rtgs)

                # reset episode-specific variables
                L = env.maze.objects.free.positions
                maze_config["start_idx"] = [L[np.random.randint(0, len(L))]]
                _, done, ep_rews = env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_acts) > batch_size:
                    break

        # Train the actor after the batch is collected
        if self.exploration is False:
            actor.train_actor(
                batch_behs,
                batch_acts,
                batch_weights,
                gvfs.num_primitives,
                actor_optimizer,
            )

        return np.mean(batch_rets)
