Model requirements
================================================================
**Note**: ``...`` means one or more dimensions, with non-zero sizes.


DQN, DQN with Per, DQN-Apex
----------------------------------------------------------------
For :class:`.DQN`, :class:`.DQN_PER`, :class:`.DQNApex`,
Machin expects a Q network accepting one or more state tensors of
shape ``[batch_size, ...]`` and returns a value tensor of
shape ``[batch_size, action_num]``, an example of the Q network is::

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

2. For :class:`.RAINBOW`, Machin expects a distributional Q network
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

3. For :class:`.DDPG`, :class:`.HDDPG`, :class:`.`