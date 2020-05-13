from .env_walker_ddpg import t, nn, Actor


class Critic(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim):
        super(Critic, self).__init__()
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        st_dim = state_dim * agent_num
        act_dim = action_dim * agent_num

        self.fc1 = nn.Linear(st_dim + act_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    # obs: batch_size * obs_dim
    def forward(self, all_states, all_actions):
        all_actions = t.flatten(all_actions, 1, -1)
        all_states = t.flatten(all_states, 1, -1)
        q = t.relu(self.fc1(t.cat((all_states, all_actions), dim=1)))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q