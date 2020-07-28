Algorithm model requirements
================================================================
Machin relies on the correct model implementation to function correctly,
different RL algorithms may need drastically dissimilar models. Therefore,
in this section, we are going to outline the detailed requirements on models
of different frameworks.

We should define some basic symbols to simplify the description:

1. ``...`` means one or more dimensions, with non-zero sizes.

Value based methods
----------------------------------------------------------------

DQN, DQNPer, DQNApex
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
For :class:`.DQN`, :class:`.DQNPer`, :class:`.DQNApex`,
Machin expects a Q network accepting one or more state tensors of
shape ``[batch_size, ...]`` and returns a value tensor of
shape ``[batch_size, action_num]``, an example of the Q network with multi
state input is::

    class QNet(nn.Module):
        def __init__(self, state_dim, action_num):
            super(QNet, self).__init__()

            self.fc1 = nn.Linear(state_dim, 16)
            self.fc2 = nn.Linear(16, 16)
            self.fc3 = nn.Linear(16, action_num)

        def forward(self, state, state2):
            state = t.cat([state, state2], dim=1)
            a = t.relu(self.fc1(state))
            a = t.relu(self.fc2(a))
            return self.fc3(a)

Normally you just need one state::

        def forward(self, state):
            a = t.relu(self.fc1(state))
            a = t.relu(self.fc2(a))
            return self.fc3(a)

Dueling DQN
*******************************
An example of the dueling DQN::

    class QNet(nn.Module):
        def __init__(self, state_dim, action_num):
            super(QNet, self).__init__()
            self.action_num = action_num
            self.fc1 = nn.Linear(state_dim, 16)
            self.fc2 = nn.Linear(16, 16)
            self.fc_adv = nn.Linear(16, action_num)
            self.fc_val = nn.Linear(16, 1)

        def forward(self, some_state):
            a = t.relu(self.fc1(some_state))
            a = t.relu(self.fc2(a))
            batch_size = a.shape[0]
            adv = self.fc_adv(a)
            val = self.fc_val(a).expand(batch_size, self.action_num)
            return val + adv - adv.mean(1, keepdim=True)


RAINBOW
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
For :class:`.RAINBOW`, Machin expects a distributional Q network
accepting one or more state tensors of shape ``[batch_size, ...]`` and
returns a value distribution tensor of shape ``[batch_size, action_num, atom_num]``,
where ``atom_num`` is the number of segments reward is divided into. The last dimension
is the probability distribution histogram of the reward, its sum should be 1.

An example of the distributional Q network is::

    class QNet(nn.Module):
        def __init__(self, state_dim, action_num, atom_num=10):
            super(QNet, self).__init__()

            self.fc1 = nn.Linear(state_dim, 16)
            self.fc2 = nn.Linear(16, 16)
            self.fc3 = nn.Linear(16, action_num * atom_num)
            self.action_num = action_num
            self.atom_num = atom_num

        def forward(self, state, state2):
            state = t.cat([state, state2], dim=1)
            a = t.relu(self.fc1(state))
            a = t.relu(self.fc2(a))
            return t.softmax(self.fc3(a)
                             .view(-1, self.action_num, self.atom_num),
                             dim=-1)

Deterministic policy gradient methods
----------------------------------------------------------------

DDPG, DDPGPer, HDDPG, TD3
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
For :class:`.DDPG`, :class:`.DDPGPer`, :class:`.HDDPG`, :class:`.TD3`, Machin
expects multiple actor and critic networks. The actor network should accept one
or more state tensors of shape ``[batch_size, ...]``