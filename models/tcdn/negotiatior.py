import torch as t
import torch.nn as nn
from models.base.tcdnnet import TCDNNet

from typing import Union


class SwarmNegotiator(nn.Module):
    def __init__(self, observe_dim, action_dim, history_depth, contiguous=True, device="cuda:0"):
        super(SwarmNegotiator, self).__init__()
        in_dim = observe_dim + action_dim + 1
        self.add_module("net", TCDNNet(in_dim, action_dim, history_depth + 1, additional_length=1,
                                       final_process="tanh" if contiguous else "softmax", device=device))
        self.device = device
        self.observe_dim = observe_dim
        self.action_dim = action_dim

    def forward(self,
                observation: t.Tensor,
                neighbor_observation: Union[t.Tensor, None],
                action: t.Tensor,
                neighbor_action: Union[t.Tensor, None],
                history: Union[t.Tensor, None],
                history_time_steps: Union[t.Tensor, None],
                time_step: int,
                rate: float):

        if neighbor_observation is not None:
            full_observe = t.cat((t.unsqueeze(observation, dim=1), neighbor_observation), dim=1)
            full_action = t.cat((t.unsqueeze(action, dim=1), neighbor_action), dim=1)
        else:
            full_observe = t.unsqueeze(observation, dim=1)
            full_action = t.unsqueeze(action, dim=1)

        # observe and action are known, reward in current step is unknown, set to 0
        curr_state = t.zeros((full_observe.shape[0], full_observe.shape[1],
                              self.observe_dim + self.action_dim + 1),
                             dtype=full_observe.dtype, device=self.device)
        curr_state[:, :, :self.observe_dim] = full_observe.to(self.device)
        curr_state[:, :, self.observe_dim: self.observe_dim + self.action_dim] = full_action.to(self.device)

        batch_size = full_observe.shape[0]
        rate_vec = t.full((batch_size, 1), rate, dtype=t.float32, device=self.device)
        time_steps = t.full((batch_size, full_observe.shape[1]), time_step,
                            dtype=t.int32, device=self.device)

        if history is not None:
            curr_state = t.cat((history, curr_state), dim=1)
            time_steps = t.cat((history_time_steps, time_steps), dim=1)

        return self.net(curr_state,
                        time_steps=time_steps,
                        additional=rate_vec)[0]
