Internal implementations
--------------------------------
In this section, we will take a brief look at the internal implementations of the
DQN framework, to better understand how to use it.

Replay memory
++++++++++++++++++++++++++++++++
Replay memory is the second core part of the whole DQN framework, Machin provides
a sophisticated but clear implementation of replay memory, to accommodate the needs
of different frameworks.

**Transition**

in order to understand how it works, we should take a step back and reexamine the
process of a MDP (Markov Decision Process). A MDP process could be described as a
chain of **transition steps**.

.. figure:: /static/tutorials/your_first_program/mdp.svg
   :alt: MDP

   MDP (Markov Decision Process)

In Machin, we store each transition step as a :class:`.TransitionBase` object, this
class manages all data of a user defined transition step, by categorizing data into
three types: major attribute, sub attribute and custom attribute.

1. Major attribute is a dictionary of tensors, mainly used to describe complex state and
action information.
2. Sub attributes are scalars or concatenatable tensors, mainly used to store less complex
states such as reward, terminal status, etc.
3. Custom attributes are not concatenatable, they could be used to store custom data
structures describing environmental specific states, that **does not have tensors** inside.

the default transition implementation is :class:`.Transition`, which have 5 attributes:

1. state (major attribute)
2. action (major attribute)
3. next_state (major attribute)
4. reward (sub attribute)
5. terminal (sub attribute)

Now that we have a very general transition data structure, which supports storing:
1. complex state information, such as visual(RGB-D), audio, physical(position, velocity, etc.),
   internal states of recurrent networks, etc.
2. complex action information, whether discreet or contiguous, single space or a combination
   of multitude of spaces, by storing them in different keys of the dictionary.
3. complex reward, whether scalar reward or vectorized reward.

We may use the stored samples to train your q network.

**Sample**
Sampling is the first step performed in the DQN framework (and almost every other frameworks),
it looks like::
    batch_size, (state, action, reward, next_state, terminal, others) = \
            self.replay_buffer.sample_batch(self.batch_size,
                                            concatenate_samples,
                                            sample_method="random_unique",
                                            sample_attrs=[
                                                "state", "action",
                                                "reward", "next_state",
                                                "terminal", "*"
                                            ])
What secret actions do this segment of code performs internally? Well, nothing
other than "sampling" and "concatenation". Argument ``sample_method`` indicates
the sample selection method implementation used in buffer, ``sample_attrs`` indicates which
attributes of each sample we are going to acquire, "*" is a wildcard selector picking
up all unsampled attributes.

Then what does "concatenation" mean? To put it simply, it will only affect "major attributes"
and "sub attributes" of each sample, nothing explains this process better than a graph: