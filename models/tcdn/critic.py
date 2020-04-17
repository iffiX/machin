import torch as t
import torch.nn as nn
from typing import Union
from models.base.tcdnnet import TCDNNet


class SwarmCritic(nn.Module):
    def __init__(self, observe_dim, action_dim, history_depth, neighbor_num, device="cuda:0"):
        super(SwarmCritic, self).__init__()
        in_dim = (observe_dim + action_dim + 1) * (1 + neighbor_num)
        self.add_module("net", TCDNNet(in_dim, 1, history_depth + 1,
                                       final_process=None, device=device))
        self.observe_dim = observe_dim
        self.action_dim = action_dim
        self.neighbor_num = neighbor_num

    def forward(self,
                observation: t.Tensor,
                neighbor_observation: t.Tensor,
                action: t.Tensor,
                neighbor_action: t.Tensor,
                history: Union[t.Tensor, None]):
        full_observe = t.cat((t.unsqueeze(observation, dim=1), neighbor_observation), dim=1)
        full_action = t.cat((t.unsqueeze(action, dim=1), neighbor_action), dim=1)
        # observe and action are known, reward in current step is unknown, set to 0
        curr_state = t.zeros((full_observe.shape[0], 1, full_observe.shape[1], history.shape[-1]),
                             dtype=full_observe.dtype, device=full_observe.device)
        curr_state[:, :, :, :self.observe_dim] = full_observe
        curr_state[:, :, :, self.observe_dim: self.observe_dim + self.action_dim] = full_action

        if history is None:
            return self.net(curr_state)[0]
        else:
            return self.net(t.cat([history, curr_state], dim=1))[0]


class WrappedCriticNet(nn.Module):
    def __init__(self, critic):
        super(WrappedCriticNet, self).__init__()
        self.add_module("critic", critic)

    def forward(self, observation, neighbor_observation, action, neighbor_action_all, history):
        return self.critic.forward(observation, neighbor_observation, action,
                                   neighbor_action_all[-1], history)
