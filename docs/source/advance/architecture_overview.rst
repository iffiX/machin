Architecture overview
================================
In this section, we will take a brief look at the internal organization of
the Machin library, to better understand the functionality of every module.

Env
--------------------------------
Currently :mod:`machin.env` has two sub modules: :mod:`machin.env.utils` and
:mod:`machin.env.wrappers`.

The submodule :mod:`machin.env.utils` of the environment module provides
some convenient utility functions you might will need in your own application,
such as disabling the rendering window while keeping the rendered result in OpenAI gym.

The submodule :mod:`machin.env.wrappers` provides process-level parallel environment
wrappers for different environments.

Framework
--------------------------------
:mod:`machin.frame` is the most important core part of the Machin library,
the framework module constitutes of:

1. :mod:`machin.frame.algorithms` : RL algorithm implementations.
2. :mod:`machin.frame.buffers` : Replay buffer implementations.
3. :mod:`machin.frame.helpers` : Utility to help you initialize a framework.
4. :mod:`machin.frame.noise` : Action space & parameter space noise generators.

Model
--------------------------------
:mod:`machin.model` is a collection of popular network models you might will use in your own
program, for example, `ResNet <https://arxiv.org/abs/1512.03385>`_.

Model module also contains the basis of all network modules: :class:`NeuralNetworkModule`,
this wrapper is built upon regular `torch.nn.Module`, and allows users to specify input/output
sub module.

Parallel
--------------------------------
:mod:`machin.parallel` is the second core part of the Machin library,
the parallel module is a collection of refined implementations including:

1. :mod:`machin.parallel.thread` : Thread (With exception catching).
2. :mod:`machin.parallel.process` : Process (With remote exception catching).
3. :mod:`machin.parallel.queues` : Queues. (Used in pools).
4. | :mod:`machin.parallel.pool` : Process pools (allow local functions,
   | customize serialization policy), thread pools, pools with contexts, etc.
5. :mod:`machin.parallel.assigner` : Heuristic based model-device assignment.
6. | :mod:`machin.parallel.server` : Implementations of different servers used in distributed
   | algorithms such as :class:`.A3C`, :class:`.DQNApex`, :class:`DDPGApex` and :class:`.IMPALA`.
7. | :mod:`machin.parallel.distributed` : A naive implementation of a part of
   | `RFC #41546 <https://github.com/pytorch/pytorch/issues/41546>`_

Utils
--------------------------------
:mod:`machin.utils` is a **messy hotchpotch** of various tools, it is very hard to categorize them,
but they could be helpful sometimes, so we left them here:

1. | :mod:`machin.utils.checker` : A checker implementation, using forward & backward hooks
   | provided by pytorch to check the input/ouput, input gradient of models. Supports user
   | defined checkers and tensorboard.
2. | :mod:`machin.utils.conf` : Functions designed to load/save a json configuration file, as
   | well as loading parametrs from commandline.
3. | :mod:`machin.utils.helper_classes` : Various helper classes, such as :class:`.Timer`, :class:`.Counter`, etc.
4. | :mod:`machin.utils.learning_rate` : Functions used in learning rate schedulers. Useful
   | if you would like to have finer control over the learning rate.
5. :mod:`machin.utils.loading` : Logging utility module.
6. | :mod:`machin.utils.media` : Media writing utility, mainly images and videos, useful if you would
   | like to log rendered environments.
7. | :mod:`machin.utils.prepare` : Functions used to create directories, loading models (take care of
   | devices automatically), for preparing a training session.
8. | :mod:`machin.utils.save_env` : A standard reinforcement training environment creator, will create
   | unique directories by time for you.
9. | :mod:`machin.utils.visualize` : Visualize your model, currently only contains some simple functions
   | for gradient flow checking.
10. :mod:`machin.utils.tensorboard`: A simple tensorboard wrapper.

