Currently, Machin supports three major types of model-free RL algorithms:

1. Value based algorithms
2. Deterministic policy based algorithms
3. Stochastic policy based algorithms
Algorithms could be grouped into respective categories with the following graph:

.. figure:: ../static/tutorials/data_flow_in_machin/category.png
   :alt: algorithm_categories

   Algorithm categories

In order to provide a consistent api interface across all algorithms, and
minimize abstractions, Machin choose to place restrictions on the output format
of users' models. The restrictions are different according to the category of
the selected RL algorithm, these restrictions **could be changed/removed** if
users decides to inherit from the default implementation and alternate the
behavior of the ``update`` function. For example, the :class:`.DDPG` algorithm
uses :meth:`.DDPG.reward_func` to calculate the reward::

    @staticmethod
    def reward_function(reward, discount, next_value, terminal, _):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * ~terminal * next_value

This function is designed for ``reward``, ``next_value``, ``terminal`` of shape
``[batch_size, 1]``, and a ``float`` discount. Users might want to implement a
vectorized reward function, which returns a reward of shape ``[batch_size, reward_dims]``,
then they will have to overload :meth:`.DDPG.reward_func` and make sure that other
statements in :meth:`.DDPG.update` will cooperate with their new reward function,
defined in their sub classes inherited from :class:`.DDPG`.

For a detailed list of these restrictions, please refer to
:ref:`Model requirements <model_requirements>` .

APIs provided by algorithms
****************************************************************
All algorithms provide three core APIs:

1. Acting API, beginning with "act".
2. Storing API, beginning with "store".
3. Training API, with name "update"

Users will invoke the "act*" api provided by the framework during sampling,
to let their models produce an action with respect to their state input,
"*" indicates additional extensions such as "_with_noise", "_discreet", etc.
depending on the implementation and type of the RL framework.

Below is a list of supported acting APIs of different frameworks:

+-----------------+-------------------------+---------------------------------------------+---------------------+-----------------------+
| Algorithm class | Acting API              | Input & output                              | Discreet/Contiguous | Note                  |
+-----------------+-------------------------+---------------------------------------------+---------------------+-----------------------+
| | DQN           | act_discreet            | | Dict[str, State[batch_size, ...]]         | D                   |                       |
| | DQNPer        |                         | | -> Action[batch_size, 1]                  |                     |                       |
| | DQNApex       +-------------------------+---------------------------------------------+---------------------+-----------------------+
| | RAINBOW       | act_discreet_with_noise | | Dict[str, State[batch_size, ...]]         | D                   |                       |
|                 |                         | | -> Action[batch_size, 1]                  |                     |                       |
+-----------------+-------------------------+---------------------------------------------+---------------------+-----------------------+
| | DDPG          | act                     | | Dict[str, State[batch_size, ...]]         | C                   |                       |
| | DDPGPer       |                         | | -> Action[batch_size, action_dim]         |                     |                       |
| | HDDPG         +-------------------------+---------------------------------------------+---------------------+-----------------------+
| | TD3           | act_with_noise          | | Dict[str, State[batch_size, ...]]         | C                   |                       |
|                 |                         | | -> Action[batch_size, action_dim]         |                     |                       |
|                 +-------------------------+---------------------------------------------+---------------------+-----------------------+
|                 | act_discreet            | | Dict[str, State[batch_size, ...]]         | D                   |                       |
|                 |                         | | -> Action[batch_size, 1]                  |                     |                       |
|                 +-------------------------+---------------------------------------------+---------------------+-----------------------+
|                 | act_discreet_with_noise | | Dict[str, State[batch_size, ...]]         | D                   |                       |
|                 |                         | | -> Action[batch_size, 1]                  |                     |                       |
+-----------------+-------------------------+---------------------------------------------+---------------------+-----------------------+
| | A2C           | act                     | | Dict[str, State[batch_size, ...]] ->      | C/D                 | | Contiguous/Discreet |
| | A3C           |                         |                                             |                     | | depends on the      |
| | PPO           |                         | | Action[batch_size, ...],                  |                     | | distribution you    |
| | IMPALA        |                         | | Log_Prob[batch_size, 1],                  |                     | | are using to        |
| | SAC           |                         | | Entropy[batch_size, 1]                    |                     | | reparameterize      |
|                 |                         |                                             |                     | | your network        |
+-----------------+-------------------------+---------------------------------------------+---------------------+-----------------------+
| MADDPG          | act                     | | List[Dict[str, State[batch_size, ...]]]   | C                   |                       |
|                 |                         | | -> List[Action[batch_size, action_dim]]   |                     |                       |
|                 +-------------------------+---------------------------------------------+---------------------+-----------------------+
|                 | act_with_noise          | | List[Dict[str, State[batch_size, ...]]]   | C                   |                       |
|                 |                         | | -> List[Action[batch_size, action_dim]]   |                     |                       |
|                 +-------------------------+---------------------------------------------+---------------------+-----------------------+
|                 | act_discreet            | | List[Dict[str, State[batch_size, ...]]]   | D                   |                       |
|                 |                         | | -> List[Action[batch_size, 1]]            |                     |                       |
|                 +-------------------------+---------------------------------------------+---------------------+-----------------------+
|                 | act_discreet_with_noise | | List[Dict[str, State[batch_size, ...]]]   | D                   |                       |
|                 |                         | | -> List[Action[batch_size, 1]]            |                     |                       |
+-----------------+-------------------------+---------------------------------------------+---------------------+-----------------------+

Algorithms generally encapsulates a replay buffer inside, the replay buffer is not
necessarily a "real" replay buffer. For online algorithms such as A2C and PPO with
no replaying mechanisms, the replay buffer is used as a place to put all of the
samples, and is cleared after every training/update step::

    # sample a batch
    batch_size, (state, action, reward, next_state,
                 terminal, target_value, advantage) = \
        self.replay_buffer.sample_batch(-1,
                                        sample_method="all",
                                        ...)

    ...
    self.replay_buffer.clear()

Most frameworks supports storing a single transition step of a MDP process, or
storing the whole MDP process at once::

    some_framework.store_transition(transition: Union[Transition, Dict])
    some_framework.store_episode(episode: List[Union[Transition, Dict]])

However, some frameworks may only support the latter one of these two APIs (Eg: IMPALA),
due to the special sampling requirements of the algorithm.

Below is a list of supported storing APIs of different frameworks:

+-----------------+--------------------------------+---------------------------------+
| Algorithm class | Storing API                    | Note                            |
+-----------------+--------------------------------+---------------------------------+
| | DQN           | store_transition/store_episode |                                 |
| | DQNPer        |                                |                                 |
| | DQNApex       |                                |                                 |
| | DDPG          |                                |                                 |
| | DDPGPer       |                                |                                 |
| | DDPGApex      |                                |                                 |
| | HDDPG         |                                |                                 |
| | TD3           |                                |                                 |
| | SAC           |                                |                                 |
+-----------------+--------------------------------+---------------------------------+
| | MADDPG        | store_transition/store_episode | | Requires you to store         |
|                 |                                | | transitions/episodes          |
|                 |                                | | of all agents at the          |
|                 |                                | | same time.                    |
+-----------------+--------------------------------+---------------------------------+
| | RAINBOW       | store_transition/store_episode | | ``store_transition`` requires |
|                 |                                | | you to calculate the n-step   |
|                 |                                | | value manually.               |
+-----------------+--------------------------------+---------------------------------+
| | A2C           | store_transition/store_episode | | ``store_transition`` requires |
| | PPO           |                                | | you to calculate the n-step   |
| | A3C           |                                | | value, and the generalized    |
|                 |                                | | advantage estimation (GAE)    |
|                 |                                | | manually.                     |
+-----------------+--------------------------------+---------------------------------+
| | IMPALA        | store_episode                  |                                 |
+-----------------+--------------------------------+---------------------------------+

All frameworks supports the ``update`` function, but the keyword arguments
of the ``update`` function might be a little bit different. For example, DDPG
allows you to choose update actor/critic/their targets, individually, while
DQN only supports choose to update Q network/its target individually.

Moreover, the update function of offline algorithms such as DDPG and online
algorithms such as A2C and PPO are different. Because A2C and PPO will not
update on outdated samples, their ``update`` function contains an internal
update loop::

    # DDPG update:
    if episode > 100:
    for i in range(step.get()):
        ddpg.update()

    # PPO update:
    # update() already contains a loop
    ppo.store_episode(tmp_observations)
    ppo.update()


and their ``update`` will also clear the internal replay buffer
every time. So you are recommended to **read the implementation** of your
selected algorithm before using it somewhere.
