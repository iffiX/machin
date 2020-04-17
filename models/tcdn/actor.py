import torch as t
import torch.nn as nn
from typing import Union
from torch.nn.functional import tanh
from models.base.tcdnnet import TCDNNet


class SwarmActor(nn.Module):
    def __init__(self, observe_dim, action_dim, history_depth, neighbor_num, contiguous=False, device="cuda:0"):
        super(SwarmActor, self).__init__()
        in_dim = (observe_dim + action_dim + 1) * (1 + neighbor_num)
        self.add_module("net", TCDNNet(in_dim, action_dim, history_depth + 1,
                                       final_process="tanh" if contiguous else "softmax", device=device))
        self.observe_dim = observe_dim
        self.neighbor_num = neighbor_num

    def forward(self,
                observation: t.Tensor,
                neighbor_observation: t.Tensor,
                history: Union[t.Tensor, None]):
        # TODO: clip observe with small attention weight to 0
        full_observe = t.cat((t.unsqueeze(observation, dim=1), neighbor_observation), dim=1)

        # observe is known, action and reward in current step is unknown, set to 0
        curr_state = t.zeros((full_observe.shape[0], 1, full_observe.shape[1], history.shape[-1]),
                             dtype=full_observe.dtype, device=full_observe.device)
        curr_state[:, :, :, :self.observe_dim] = full_observe

        if history is None:
            return self.net(curr_state)[0]
        else:
            return self.net(t.cat([history, curr_state], dim=1))[0]


class WrappedActorNet(nn.Module):
    def __init__(self, base_actor, negotiator):
        super(WrappedActorNet, self).__init__()

        self.add_module("actor", base_actor)
        self.add_module("negotiator", negotiator)

    def forward(self,
                observation, neighbor_observation,
                neighbor_action_all, negotiate_rate_all, history):
        # used in ddpg training, input is a sample from the replay buffer
        # currently, only support sample batch size = 1
        action = self.actor(observation, neighbor_observation, history)
        for nego_rate, neigh_action in zip(negotiate_rate_all, neighbor_action_all[:-1]):
            action += nego_rate * \
                      self.negotiator(observation, neighbor_observation, history,
                                      action, neigh_action, nego_rate)
        return tanh(action)
