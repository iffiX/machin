Recurrent networks
================================================================
**Author**: `Muhan Li <https://github.com/iffiX>`_

**Full code 1**: `DRQN <https://github.com/iffiX/machin/blob/master/examples/tutorials/recurrent_networks/drqn.py>`_
**Full code 2**: `DQN <https://github.com/iffiX/machin/blob/master/examples/tutorials/recurrent_networks/dqn.py>`_

Preface
----------------------------------------------------------------
In this tutorial, we are going to try and implement the "DRQN" architecture
described in `Deep Recurrent Q-Learning for Partially Observable MDPs <https://arxiv.org/pdf/1507.06527.pdf>`_

Now, in order to implement the "DRQN" architecture, we should have a solid grasp of
the following related aspects in advance:

1. `DQN framework <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>`_
2. `Recurrent neural networks <https://en.wikipedia.org/wiki/Recurrent_neural_network>`_, `LSTM <https://www.bioinf.jku.at/publications/older/2604.pdf>`_ and `GRU <https://arxiv.org/pdf/1412.3555>`_
3. `MDP <https://en.wikipedia.org/wiki/Markov_decision_process>`_ and `POMDP <https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process>`_

Recurrent network are introduced into the reinforcement learning field to deal with POMDP models, in which
agents are not able to observe the full state of the environment, and they have to rely on their internal
memories of their past observations. There are two ways to achieve this target, namely:

1. Put past observations into a memory region, then pass the whole memory region to a non-recurrent network.
2. Use recurrent networks, and use the hidden states to pass past information to the next step.

The `essay <https://arxiv.org/pdf/1507.06527.pdf>`_ used Atari games as the benchmark suite, they compared DRQN (method #2)
with DQN (method #1) in multiple scenarios, and shows that DRQN has significant advantage over DQN in the `frostbite <https://gym.openai.com/envs/Frostbite-v0/>`_
game, while performing about as good as / fail to compete with DQN in many other Atari games. Therefore, in this tutorial,
we are going to "replicate" their result, and use `GRU <https://arxiv.org/pdf/1412.3555>`_ instead of `LSTM <https://www.bioinf.jku.at/publications/older/2604.pdf>`_
to speed up model inference and training.

Network architecture
----------------------------------------------------------------
We are going to use exactly the same DRQN architecture are the original essay does,
while **replacing the LSTM layer with GRU layer**:

.. figure:: ../static/tutorials/recurrent_networks/drqn.png
   :alt: drqn

   Original DRQN

Training method
----------------------------------------------------------------
Authors of the original paper choose to train the LSTM layer along with the CNN layers,
in order to deal with the "hidden state" input of the LSTM layer, they proposed two methods:

1. Bootstrapped sequential updates
2. Bootstrapped random updates

We can describe these two methods with the following graph:

.. figure:: ../static/tutorials/recurrent_networks/drqn_training.svg
   :alt: drqn_training

   DRQN training

Sequential updates use the hidden state from the previous update, while random updates use
a zeroed hidden state, therefore decouple consecutive updates. Authors of DRQN says that these
two methods are:

    Experiments indicate that both types of updates are viable and yield convergent policies with similar performance
    across a set of games. Therefore, to limit complexity, all results herein use the randomized update strategy.

Therefore we are also going to use the second training method. Currently, the :class:`DQN` framework doesn't
support method #1, because the hidden states need to be returned and passed to the next ``update`` call, you
may implement this method yourself, though.

Setup
----------------------------------------------------------------
In this section, we are going to:

1. Define DRQN and DQN network architecture.
2. Use the :class:`DQN` framework to train these two architectures individually.
3. Compare their results.

Network definitions
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Network definitions of DQN and DRQN are as follows:

DQN::


DRQN::