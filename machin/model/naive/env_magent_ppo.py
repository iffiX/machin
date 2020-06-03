import numpy as np
import torch as t
import torch.nn as nn

from torch.distributions import Categorical


class Actor(nn.Module):
    # naive actor for env.magent
    kernel_num = [32, 32]
    view_embed_size = 256

    def __init__(self, view_shape, feature_dim, action_dim, use_conv=True):
        super(Actor, self).__init__()

        # feature shape: (view_width, view_height, n_channel)
        # view shape: (ID embedding + last action + last reward + relative pos)

        self.use_conv = use_conv
        hidden_size = self.view_embed_size + feature_dim

        if use_conv:
            self.conv1 = nn.Conv2d(in_channels=view_shape[-1],
                                   out_channels=self.kernel_num[0],
                                   kernel_size=3)
            self.conv2 = nn.Conv2d(in_channels=self.kernel_num[0],
                                   out_channels=self.kernel_num[1],
                                   kernel_size=3)
            new_h = view_shape[0] - 4
            new_w = view_shape[1] - 4
            self.fc1 = nn.Linear(new_h * new_w * self.kernel_num[1], self.view_embed_size)

        else:
            self.fc1 = nn.Linear(np.prod(view_shape), self.view_embed_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, view, feature, action=None):
        if self.use_conv:
            v = t.relu(self.conv1(view))
            v = t.relu(self.conv2(v))
            v = v.flatten(start_dim=1)
        else:
            v = t.relu(self.fc1(view.flatten(start_dim=1)))

        v = t.relu(self.fc1(v))
        a = t.cat([v, feature], dim=1)
        a = t.relu(self.fc2(a))
        a = t.softmax(self.fc3(a), dim=1)

        a_dist = Categorical(probs=a)
        action = action.squeeze(1) if action is not None else a_dist.sample()
        entropy = a_dist.entropy()
        return action.unsqueeze(1).detach(), \
               a_dist.log_prob(action).unsqueeze(1), \
               entropy.unsqueeze(1)


class Critic(nn.Module):
    # naive actor for env.magent
    kernel_num = [32, 32]
    view_embed_size = 256

    def __init__(self, view_shape, feature_dim, use_conv=True):
        super(Critic, self).__init__()

        # feature shape: (view_width, view_height, n_channel)
        # view shape: (ID embedding + last action + last reward + relative pos)

        self.use_conv = use_conv
        hidden_size = self.view_embed_size + feature_dim

        if use_conv:
            self.conv1 = nn.Conv2d(in_channels=view_shape[-1],
                                   out_channels=self.kernel_num[0],
                                   kernel_size=3)
            self.conv2 = nn.Conv2d(in_channels=self.kernel_num[0],
                                   out_channels=self.kernel_num[1],
                                   kernel_size=3)
            new_h = view_shape[0] - 4
            new_w = view_shape[1] - 4
            self.fc1 = nn.Linear(new_h * new_w * self.kernel_num[1], self.view_embed_size)

        else:
            self.fc1 = nn.Linear(np.prod(view_shape), self.view_embed_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, view, feature):
        if self.use_conv:
            v = t.relu(self.conv1(view))
            v = t.relu(self.conv2(v))
            v = v.flatten(start_dim=1)
        else:
            v = t.relu(self.fc1(view.flatten(start_dim=1)))

        v = t.relu(self.fc1(v))
        a = t.cat([v, feature], dim=1)
        a = t.relu(self.fc2(a))
        a = self.fc3(a)

        return a