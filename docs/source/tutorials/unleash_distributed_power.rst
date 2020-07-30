Unleash distributed power
================================================================
**Author**: `Muhan Li <https://github.com/iffiX>`_

In this tutorial, we are going to try out some distributed single agent RL algorithms,
they are:

1. :class:`.A3C`
2. :class:`.DQNApex` and :class:`.DDPGApex`
3. :class:`.IMPALA`

Now let's begin!

A3C
----------------------------------------------------------------

**Full code**: `Github <https://github.com/iffiX/machin/blob/master/examples/tutorials/unleash_distributed_power/a3c.py>`_

A3C is the simplest distributed RL algorithm, among them all. We can describe its
implementation with the following graph:

.. figure:: ../static/tutorials/unleash_distributed_power/a3c.svg
   :alt: a3c

   A3C architecture

And a segment of pseudo code:

.. figure:: ../static/tutorials/unleash_distributed_power/a3c_pcode.png
   :alt: a3c_pcode

   A3C pesudo code

A3C is basically a bunch of A2C agents with a gradient reduction server. A3C(A2C)
agents will interact with their environment simulators, train their local actors
and critics, then push gradients to the gradient reduction server, the gradient
reduction server will apply reduced gradients to its internal models (managed actor
and critic network), then push the updated parameters to a key-value server. Agents
will be able to pull the newest parameters and continue updating.

All A3C agents are fully asynchronous, gradient pushing & parameter pulling are asynchronous
as well.

We will use the "CartPole-v0" environment from OpenAI Gym as an example, the actor network
and critic network are as follows::

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

In order to initialize the :class:`.A3C` framework, we need to provide a
:class:`PushPullGradServer` to it, Machin provides some helpful utility functions
to aid inexperienced users initialize the distributed environment easily::

    from machin.frame.helpers.servers import grad_server_helper
    servers = grad_server_helper(
        lambda: Actor(observe_dim, action_num),
        lambda: Critic(observe_dim),
        learning_rate=5e-3
    )

**Note** all helpers from :mod:`machin.frame.helpers.servers` requires all
processes in the distributed