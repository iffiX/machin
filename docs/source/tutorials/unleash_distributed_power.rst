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
processes in the distributed world to enter.

Then we can initialize the distributed world as::

    actor = Actor(observe_dim, action_num)
    critic = Critic(observe_dim)

    # in all test scenarios, all processes will be used as reducers
    servers = grad_server_helper(
        lambda: Actor(observe_dim, action_num),
        lambda: Critic(observe_dim),
        learning_rate=5e-3
    )
    a3c = A3C(actor, critic,
              nn.MSELoss(reduction='sum'),
              servers)

And start training, just as the A2C algorithm::

    # manually control syncing to improve performance
    a3c.set_sync(False)

    # begin training
    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    terminal = False

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0

        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

        # manually pull the newest parameters
        a3c.manual_sync()
        tmp_observations = []
        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = a3c.act({"state": old_state})[0]
                state, reward, terminal, _ = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward

                tmp_observations.append({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps
                })

        # update
        a3c.store_episode(tmp_observations)
        a3c.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)
        logger.info("Process {} Episode {} total reward={:.2f}"
                    .format(rank, episode, smoothed_total_reward))

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                # will cause torch RPC to complain
                # since other processes may have not finished yet.
                # just for demonstration.
                exit(0)
        else:
            reward_fulfilled = 0

A3C agents should will be successfully trained within about 1500 episodes,
they converge must slower than A2C agents::

    [2020-07-31 00:21:37,690] <INFO>:default_logger:Process 1 Episode 1346 total reward=184.91
    [2020-07-31 00:21:37,723] <INFO>:default_logger:Process 0 Episode 1366 total reward=171.22
    [2020-07-31 00:21:37,813] <INFO>:default_logger:Process 2 Episode 1345 total reward=190.73
    [2020-07-31 00:21:37,903] <INFO>:default_logger:Process 1 Episode 1347 total reward=186.41
    [2020-07-31 00:21:37,928] <INFO>:default_logger:Process 0 Episode 1367 total reward=174.10
    [2020-07-31 00:21:38,000] <INFO>:default_logger:Process 2 Episode 1346 total reward=191.66
    [2020-07-31 00:21:38,000] <INFO>:default_logger:Environment solved!


DQNApex and DDPGApex
----------------------------------------------------------------