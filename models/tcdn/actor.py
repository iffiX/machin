import torch as t
import torch.nn as nn
from typing import Union
from torch.nn.functional import tanh
from models.base.tcdnnet import TCDNNet


class SwarmActor(nn.Module):
    def __init__(self, observe_dim, action_dim, history_depth, contiguous=False, device="cuda:0"):
        super(SwarmActor, self).__init__()
        in_dim = observe_dim + action_dim + 1
        self.add_module("net", TCDNNet(in_dim, action_dim, history_depth + 1,
                                       final_process="tanh" if contiguous else "softmax", device=device))

        self.device = device
        self.observe_dim = observe_dim
        self.action_dim = action_dim
        # self.last_observe = None

    def forward(self,
                observation: t.Tensor,
                neighbor_observation: Union[t.Tensor, None],
                history: Union[t.Tensor, None],
                history_time_steps: Union[t.Tensor, None],
                time_step):
        # TODO: clip observe with small attention weight to 0
        if neighbor_observation is not None:
            full_observe = t.cat((t.unsqueeze(observation, dim=1), neighbor_observation), dim=1)
        else:
            full_observe = t.unsqueeze(observation, dim=1)

        # observe is known, action and reward in current step is unknown, set to 0
        curr_state = t.zeros((full_observe.shape[0], full_observe.shape[1],
                              self.observe_dim + self.action_dim + 1),
                             dtype=full_observe.dtype, device=self.device)
        curr_state[:, :, :self.observe_dim] = full_observe.to(self.device)


        # print("observe:")
        # print(observation)
        # if self.last_observe is not None:
        #     print("diff:")
        #     print(observation - self.last_observe)
        # self.last_observe = observation

        batch_size = full_observe.shape[0]
        time_steps = t.full((batch_size, full_observe.shape[1]), time_step,
                            dtype=t.int32, device=self.device)

        if history is not None:
            curr_state = t.cat((history, curr_state), dim=1)
            time_steps = t.cat((history_time_steps, time_steps), dim=1)

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
        # currently, only support sample batch size = 1
        action = self.actor(observation, neighbor_observation, history, history_time_steps, time_step)
        for nego_rate, neigh_action in zip(negotiate_rate_all, neighbor_action_all[:-1]):
            action += nego_rate * \
                      self.negotiator(observation, neighbor_observation,
                                      action, neigh_action, history, history_time_steps,
                                      time_step, nego_rate)
        return tanh(action)
