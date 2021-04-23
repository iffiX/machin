Algorithm model requirements
================================================================
**Author**: `Muhan Li <https://github.com/iffiX>`_

Machin relies on the correct model implementation to function correctly,
different RL algorithms may need drastically dissimilar models. Therefore,
in this section, we are going to outline the detailed requirements on models
of different frameworks.

We will use some basic symbols to simplify the model signature:

1. | ``abc_0[*]`` means a tensor with meaning "abc", and has index 0 in all argument tensors with the same meaning, "*" is a wildcard which accepts one or more non-zero dimensions, valid examples are:
   |
   | state_0[batch_size, 1]
   | state_1[1, 2, 3, 4, 5]
   | state_2[...]
2. ``...`` means one or more arguments (tensors/not tensors), or one or more dimensions, with non-zero sizes.
3. ``<>`` means optional results / arguments. ``<...>`` means any number of optional results / arguments.

**Note**: When an algorithm API returns one result, the result will not be wrapped in a tuple, when it returns multiple results, results will be wrapped in a tuple. This design is made to support::

    # your Q network model only returns a Q value tensor
    act = dqn.act({"state": some_state})

    # your Q network model returns Q value tensor with some additional hidden states
    act, h = dqn.act({"state": some_state})

**Note**: the ``forward`` method signature
**must conform to the following definitions exactly**,
with no more or less arguments/keyword arguments.

**Note**: the requirements in this document does not apply to the conditions
where: (1) you have made a custom implementation (2) you have inherited frameworks
and customized their result adaptors like :meth:`.DDPG.action_transform_function`,
etc.

DQN, DQNPer, DQNApex
----------------------------------------------------------------
For :class:`.DQN`, :class:`.DQNPer`, :class:`.DQNApex`,
Machin expects a Q network::

    QNet(state_0[batch_size, ...],
         ...,
         state_n[batch_size, ...])
    -> q_value[batch_size, action_num], <...>

where ``action_num`` is the number of available discreet actions.

Example::

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

Dueling DQN
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
------------------------------------------------------------------
For :class:`.RAINBOW`, Machin expects a distributional Q network::

    DistQNet(state_0[batch_size, ...],
             ...,
             state_n[batch_size, ...])
    -> q_value_dist[batch_size, action_num, atom_num], <...>

where:

1. ``action_num`` is the number of available discreet actions
2. ``atom_num`` is the number of q value distribution bins
3. ``sum(q_value_dist[i, j, :]) == 1``


Example::

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


DDPG, DDPGPer, DDPGApex, HDDPG, TD3
------------------------------------------------------------------
For :class:`.DDPG`, :class:`.DDPGPer`, :class:`.DDPGApex`, :class:`.HDDPG`,
:class:`.TD3`, Machin expects multiple actor and critic networks like::

    Actor(state_0[batch_size, ...],
          ...,
          state_n[batch_size, ...])
    -> action[batch_size, ...], <...>          # if contiguous
    -> action[batch_size, action_num], <...>   # if discreet

    Critic(state_0[batch_size, ...],
           ...,
           state_n[batch_size, ...],
           action[batch_size, .../action_num])
    -> q_value[batch_size, 1], <...>

where:

1. ``action_num`` is the number of available discreet actions
2. ``sum(action[i, :]) == 1`` if discreet.

Example::

    class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.action_range = action_range

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.tanh(self.fc3(a)) * self.action_range
        return a


    class ActorDiscrete(nn.Module):
        def __init__(self, state_dim, action_dim):
            # action_dim means action_num here
            super(ActorDiscrete, self).__init__()

            self.fc1 = nn.Linear(state_dim, 16)
            self.fc2 = nn.Linear(16, 16)
            self.fc3 = nn.Linear(16, action_dim)

        def forward(self, state):
            a = t.relu(self.fc1(state))
            a = t.relu(self.fc2(a))
            a = t.softmax(self.fc3(a), dim=1)
            return a


    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(Critic, self).__init__()

            self.fc1 = nn.Linear(state_dim + action_dim, 16)
            self.fc2 = nn.Linear(16, 16)
            self.fc3 = nn.Linear(16, 1)

        def forward(self, state, action):
            state_action = t.cat([state, action], 1)
            q = t.relu(self.fc1(state_action))
            q = t.relu(self.fc2(q))
            q = self.fc3(q)
            return q

A2C, PPO, TRPO, A3C, IMPALA
------------------------------------------------------------------
For :class:`.A2C`, :class:`.PPO`, :class:`.TRPO`, :class:`.A3C`, :class:`.IMPALA`,
Machin expects multiple actor and critic networks like::

    Actor(state_0[batch_size, ...],
          ...,
          state_n[batch_size, ...],
          action[batch_size, ...]=None)
    -> action[batch_size, ...], <...>
       action_log_prob[batch_size, <1>]
       distribution_entropy[batch_size, <1>]

    Critic(state_0[batch_size, ...],
           ...,
           state_n[batch_size, ...])
    -> value[batch_size, 1], <...>

where:

1. ``action`` can be sampled from pytorch distributions using non-differentiable ``sample()``.
2. ``action_log_prob`` is the log likelihood of the sampled action, must be differentiable.
3. ``distribution_entropy`` is the entropy value of reparameterized distribution, must be differentiable.
4. ``Actor`` must calculate the log probability of the input ``action`` if it is not ``None``, and return the input action **as-is**.

Example::

    class Actor(nn.Module):
        def __init__(self, state_dim, action_num):
            super(Actor, self).__init__()

            self.fc1 = nn.Linear(state_dim, 16)
            self.fc2 = nn.Linear(16, 16)
            self.fc3 = nn.Linear(16, action_num)

        def forward(self, state, action=None):
            a = t.relu(self.fc1(state))
            a = t.relu(self.fc2(a))
            probs = t.softmax(self.fc3(a), dim=1)
            dist = Categorical(probs=probs)
            act = (action
                   if action is not None
                   else dist.sample())
            act_entropy = dist.entropy()
            act_log_prob = dist.log_prob(act.flatten())
            return act, act_log_prob, act_entropy

    class ActorContinuous(nn.Module):
        def __init__(self, state_dim, action_dim, action_range):
            super(Actor, self).__init__()

            self.fc1 = nn.Linear(state_dim, 16)
            self.fc2 = nn.Linear(16, 16)
            self.mu_head = nn.Linear(16, action_dim)
            self.sigma_head = nn.Linear(16, action_dim)
            self.action_range = action_range

        def forward(self, state, action=None):
            a = t.relu(self.fc1(state))
            a = t.relu(self.fc2(a))
            mu = self.mu_head(a)
            sigma = softplus(self.sigma_head(a))
            dist = Normal(mu, sigma)
            act = (action
                   if action is not None
                   else dist.sample())
            act_entropy = dist.entropy().sum(1, keepdim=True)

            # If your distribution is different from "Normal" then you may either:
            # 1. deduce the remapping function for your distribution and clamping
            #    function such as tanh
            # 2. clamp you action, but please take care:
            #    1. do not clamp actions before calculating their log probability,
            #       because the log probability of clamped actions might will be
            #       extremely small, and will cause nan
            #    2. do not clamp actions after sampling and before storing them in
            #       the replay buffer, because during update, log probability will
            #       be re-evaluated they might also be extremely small, and network
            #       will "nan". (might happen in PPO, not in SAC because there is
            #       no re-evaluation)
            # Only clamp actions sent to the environment, this is equivalent to
            # change the action reward distribution, will not cause "nan", but
            # this makes your training environment further differ from you real
            # environment.

            # the suggested way to confine your actions within a valid range
            # is not clamping, but remapping the distribution
            # from the SAC essay:   https://arxiv.org/abs/1801.01290
            act_log_prob = dist.log_prob(act)
            act_tanh = t.tanh(act)
            act = act_tanh * self.action_range

            # the distribution remapping process used in the original essay.
            act_log_prob -= t.log(self.action_range *
                                  (1 - act_tanh.pow(2)) +
                                  1e-6)
            act_log_prob = act_log_prob.sum(1, keepdim=True)

            return act, act_log_prob, act_entropy

    class Critic(nn.Module):
        def __init__(self, state_dim):
            super(Critic, self).__init__()

            self.fc1 = nn.Linear(state_dim, 16)
            self.fc2 = nn.Linear(16, 16)
            self.fc3 = nn.Linear(16, 1)

        def forward(self, state):
            v = t.relu(self.fc1(state))
            v = t.relu(self.fc2(v))
            v = self.fc3(v)
            return v

SAC
------------------------------------------------------------------
For :class:`.SAC`, Machin expects an actor similar to the actors in stochastic
policy gradient methods such as :class:`.A2C`, and multiple critics similar to critics
used in :class:`.DDPG`::

    Actor(state_0[batch_size, ...],
          ...,
          state_n[batch_size, ...],
          action[batch_size, ...]=None)
    -> action[batch_size, ...]
       action_log_prob[batch_size, <1>]
       distribution_entropy[batch_size, <1>],
       <...>

    Critic(state_0[batch_size, ...],
           ...,
           state_n[batch_size, ...],
           action[batch_size, .../action_num])
    -> q_value[batch_size, 1], <...>

where:

1. ``action`` can only be sampled from pytorch distributions using **differentiable** ``rsample()``.
2. ``action_log_prob`` is the log likelihood of the sampled action, must be differentiable.
3. ``distribution_entropy`` is the entropy value of reparameterized distribution, must be differentiable.
4. ``Actor`` must calculate the log probability of the input ``action`` if it is not ``None``, and return the input action **as-is**.

Example::

    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, action_range):
            super(Actor, self).__init__()

            self.fc1 = nn.Linear(state_dim, 16)
            self.fc2 = nn.Linear(16, 16)
            self.mu_head = nn.Linear(16, action_dim)
            self.sigma_head = nn.Linear(16, action_dim)
            self.action_range = action_range

        def forward(self, state, action=None):
            a = t.relu(self.fc1(state))
            a = t.relu(self.fc2(a))
            mu = self.mu_head(a)
            sigma = softplus(self.sigma_head(a))
            dist = Normal(mu, sigma)
            act = (action
                   if action is not None
                   else dist.rsample())
            act_entropy = dist.entropy().sum(1, keepdim=True)

            # the suggested way to confine your actions within a valid range
            # is not clamping, but remapping the distribution
            act_log_prob = dist.log_prob(act)
            act_tanh = t.tanh(act)
            act = act_tanh * self.action_range

            # the distribution remapping process used in the original essay.
            act_log_prob -= t.log(self.action_range *
                                  (1 - act_tanh.pow(2)) +
                                  1e-6)
            act_log_prob = act_log_prob.sum(1, keepdim=True)

            return act, act_log_prob, act_entropy

    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(Critic, self).__init__()

            self.fc1 = nn.Linear(state_dim + action_dim, 16)
            self.fc2 = nn.Linear(16, 16)
            self.fc3 = nn.Linear(16, 1)

        def forward(self, state, action):
            state_action = t.cat([state, action], 1)
            q = t.relu(self.fc1(state_action))
            q = t.relu(self.fc2(q))
            q = self.fc3(q)
            return q

