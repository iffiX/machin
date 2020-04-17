import torch as t
import torch.nn as nn
from models.base.tcdnnet import TCDNNet

from typing import Union

class SwarmNegotiator(nn.Module):
    def __init__(self, observe_dim, action_dim, history_depth, neighbor_num, contiguous=True, device="cuda:0"):
        super(SwarmNegotiator, self).__init__()
        in_dim = (observe_dim + action_dim + 1) * (1 + neighbor_num)
        self.add_module("net", TCDNNet(in_dim, action_dim, history_depth + 1, additional_length=1,
                                       final_process="tanh" if contiguous else "softmax", device=device))
        self.observe_dim = observe_dim
        self.action_dim = action_dim
        self.neighbor_num = neighbor_num

    def forward(self, observe: t.Tensor, neighbor_observe: t.Tensor, history: Union[t.Tensor, None],
                action: t.Tensor, neighbor_action: t.Tensor, rate):
        full_observe = t.cat((t.unsqueeze(observe, dim=1), neighbor_observe), dim=1)
        full_action = t.cat((t.unsqueeze(action, dim=1), neighbor_action), dim=1)

        # observe and action are known, reward in current step is unknown, set to 0
        curr_state = t.zeros((full_observe.shape[0], 1, full_observe.shape[1], history.shape[-1]),
                             dtype=full_observe.dtype, device=full_observe.device)
        curr_state[:, :, :, :self.observe_dim] = full_observe
        curr_state[:, :, :, self.observe_dim: self.observe_dim + self.action_dim] = full_action

        batch_size = full_observe.shape[0]
        rate_vec = t.full([batch_size, 1], rate, dtype=full_observe.dtype, device=full_observe.device)

        if history is None:
            return self.net(curr_state, rate_vec)[0]
        else:
            return self.net(t.cat([history, curr_state], dim=1), rate_vec)[0]
