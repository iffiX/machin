import torch as t
import torch.nn as nn
from typing import Union
from torch.nn.functional import tanh
from models.base.tcdnnet import TCDNNet


class SwarmActor(nn.Module):
    def __init__(self, observe_dim, action_dim, history_depth, neighbor_num, contiguous=False, device="cuda:0"):
        super(SwarmActor, self).__init__()
        in_dim = observe_dim + action_dim + 1
        seq_length = (history_depth + 1) * (neighbor_num + 1)
        self.add_module("net", TCDNNet(in_dim, action_dim, seq_length,
                                       final_process="tanh" if contiguous else "softmax", device=device))

        self.device = device
        self.observe_dim = observe_dim
        self.action_dim = action_dim
        # self.last_observe = None

    def forward(self,
                observation: t.Tensor,
                neighbor_observation: t.Tensor,
                history: t.Tensor,
                history_time_steps: t.Tensor,
                time_step: t.Tensor):
        # TODO: clip observe with small attention weight to 0
        # shape: (B, neighbor_num+1, observation_dim)
        full_observe = t.cat((t.unsqueeze(observation, dim=1), neighbor_observation), dim=1)


        # observe is known, action and reward in current step is unknown, set to 0
        # shape: (B, neighbor_num+1, action_dim+observation_dim+reward_dim)
        curr_state = t.zeros((full_observe.shape[0], full_observe.shape[1],
                              self.observe_dim + self.action_dim + 1),
                             dtype=full_observe.dtype, device=self.device)
        curr_state[:, :, :self.observe_dim] = full_observe.to(self.device)

        # shape: (B, (neighbor_num+1) * history_depth, action_dim+observation_dim+reward_dim)
        curr_state = t.cat((history, curr_state), dim=1)
        time_steps = t.cat((history_time_steps, time_step), dim=1)

        return self.net(curr_state,
                        time_steps=time_steps)[0]


class WrappedActorNet(nn.Module):
    def __init__(self, base_actor, negotiator):
        super(WrappedActorNet, self).__init__()

        self.add_module("actor", base_actor)
        self.add_module("negotiator", negotiator)

    def forward(self,
                observation,
                neighbor_observation,
                neighbor_action_all,
                negotiate_rate_all,
                history,
                history_time_steps,
                time_step):
        # used in ddpg training, input is a sample from the replay buffer
        action = self.actor(observation, neighbor_observation, history, history_time_steps, time_step)
        for nround in range(negotiate_rate_all.shape[1]):
            action += negotiate_rate_all[:, nround].unsqueeze(-1) * \
                      self.negotiator(observation, action,
                                      neighbor_observation,
                                      neighbor_action_all[:, nround], history, history_time_steps,
                                      time_step, negotiate_rate_all[:, nround].unsqueeze(-1))
        return tanh(action)
