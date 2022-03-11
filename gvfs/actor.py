import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, num_primitives):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, action_dim * num_primitives)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return self.l4(a)

    def train_actor(
        self, batch_behs, batch_acts, batch_weights, num_primitives, actor_optimizer
    ):

        action_probs = torch.reshape(
            self.forward(torch.FloatTensor([1]).to("cpu")), (4, 4)
        )

        loss_primitive = torch.zeros(4)
        for prim in range(num_primitives):
            m = Categorical(logits=action_probs[prim])
            if np.size(np.where(np.array(batch_behs) == prim)[0]) != 0:
                index_beh = np.where(np.array(batch_behs) == prim)[0]
                loss_primitive[prim] = torch.mean(
                    -m.log_prob(
                        torch.as_tensor(
                            np.array(batch_acts)[index_beh], dtype=torch.float32
                        ).to("cpu")
                    )
                    * torch.as_tensor(
                        np.array(batch_weights)[index_beh], dtype=torch.float32
                    ).to("cpu")
                )

        # Compute actor losse
        actor_loss = torch.mean(loss_primitive)
        # Optimize the actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
