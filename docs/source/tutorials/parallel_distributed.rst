Parallel, distributed
================================================================
**Author**: `Muhan Li <https://github.com/iffiX>`_

This tutorial is going to give you a brief overview of how to write
parallel & distributed programs, with Machin.

What are they?
----------------------------------------------------------------
**Parallel** means a set of computation processes executed simultaneously,
whether synchronous or asynchronous.

**Distributed** means a system whose components are located on different
entities, which are usually computers connected by networks.

Overview
----------------------------------------------------------------

Parallel
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
**Traditional perspective**

From the perspective of traditional parallel computation, there are many levels
of parallelism, supported by Machin, based on PyTorch, from fine to coarse:

1. Element level parallelism, based on multidimensional tensor computations.
2. Task level parallelism, achieved by multi-threading, either provided by
   python threads, or the JIT fork mechanism of PyTorch (with no GIL).
3. Task level parallelism, achieved by multi-processing, either on the same
   node, or on different nodes.

For element level parallelism, we can either use existing tensor operators,
or use more flexible operators such as ``torch.einsum`` to make customized operators,
or write our own CUDA kernels. We can even use ``torch.jit`` to compile our
models and get some performance improvements over plain python APIs. Machin doesn't
provide any utility in this area.

For based task level parallelism, the basic python libraries, such as
``threading`` and ``multiprocessing`` already provide enough functions to achieve the
latter two parallelisms. Machin provides the following enhancements:

1. Watch for exceptions happening in threads/processes.
2. Process/Thread pools with local function execution ability, accurate control over tensor serialization policy.
3. Process/Thread pools with contexts, allow users to pass hard-to-construct objects before executing tasks.
4. Inter-process queues with accurate control over tensor serialization policy.

**Neural network perspective**

From the perspective of neural networks, there are some parallelism
paradigms we would like to achieve, with traditional parallel architectures:

1. Model level parallelism in small batch inference of many small models.
2. Model level parallelism in large batch inference of one potentially huge model.
3. Model level "parallelism" in storing an extremely large model across multiple devices or nodes.

Currently, there is no perfect way to deal with the first scenario, because threads
in python are constrained by GIL, while processes are too slow. In :class:`.MADDPG`,
Machin choose to utilize the JIT function provided by pytorch, and use compiled JIT
models to work around the GIL restriction, this method is proved to have about
50% speed advantage over regular thread pools.

The second scenario could be dealt with ``DistributedDataParallel`` in PyTorch, by
splitting the large batch into several smaller batches, then perform inference on
different processes asynchronously.

The last scenario is also known as "model sharding", which means split a huge model
up into several smaller models. It would be more favorable to users if this could be
done automatically by the framework. However, due to the design of PyTorch, where
tensors, not models, are real entities bound to device, it is not possible to achieve
this function directly, with PyTorch, as of version 1.5.0. Machin currently does not
provide automatic model sharding as well, but our internal implementation do support
implementing such a feature, this feature might will be added in the future. Currently,
Machin only provides automatic assignment of (splitted) models, with :class:`.ModelAssigner`.

**Reinforcement learning perspective**

When it comes to RL algorithms, these parallelisms are usually required:

1. Environment parallelism, where multiple same environments are executed synchronously in parallel, to produce larger batches of observations.
2. Agent parallelism, where multiple agents are learning synchronously or asynchronously, like :class:`.A3C`, :class:`.DQNApex`.
3. Agent parallelism in multi-agent algorithms, where multiple agents of different types are learning synchronously or asynchronously, like :class:`.MADDPG`

Machin provides parallel environment wrappers for the first scenario, like :class:`.openai_gym.ParallelWrapperSubProc`, which starts
multiple worker processes, create an environment instance in each worker, then send commands and receive responses in batches.

The second scenario is more tricky, since agents are usually distributed across
"parataxis" (same-level) processes, and on multiple nodes rather than "hypotaxis"
sub-processes started in a process pool, on the same node. We will discuss this
part in the `Distributed`_ section.

The third scenario depends on the RL algorithm framework, for :class:`MADDPG`, each agent corresponds
to a pair of separate actor and critic, in this case, only task level parallelism based threads could
be used to solve the problem, because it is hard to create batches, caused by parameter and model architecture
difference. But if we are using single agent RL algorithms such as :class:`DDPG` to train a group of
homogeneous agents, then batching is preferred due its efficientcy.

Distributed
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Distributed is awesome, as well as extremely painful to deal with, hard to design,
and even harder to debug, because applications are often required to have some
crucial features like consistency, availability, partition-tolerance, and good performance.

Currently, since Machin relies on the PyTorch RPC framework, it does not provide
any distribute mechanism able to guarantee any part of
consistency, availability or partition-tolerance, due to some limitations in
the PyTorch RPC framework, as of version 1.5.0.

What Machin provide is a more advanced set of RPC APIs: an implementation of RPC groups (namespace), on which you can
register a service with ``register`` or share a resource with ``pair``, like the code below::

        self.group.pair(server_name,
                        OrderedServerSimple(self.server_name, self.group))
        self.group.register(server_name + "/_push_service", self._push_service)
        self.group.register(server_name + "/_pull_service", self._pull_service)

This "DNS" like mechanism enables Machin to abstract away "name"s of processes, and a specific server process,
instead, every process who wants to access the service/resource are faced with a registration
table. This table could be different, depending on the actual process running the service,
and the internal implementation of the service. With this design, Machin is able to provide
some general distributed implementations such as :class:`.DistributedBuffer`, :class:`DistributedPrioritizedBuffer`,
:class:`.PushPullGradServer`, etc.

Apart from this, Machin just provides a thin layer of incapsulation over the somewhat complex
APIs of ``torch.distributed`` and ``torch.distributed.rpc``, to make them less confusing.


Examples
----------------------------------------------------------------
In order to fully understand all the functions provided :mod:`machin.parallel`, we should
read some detailed use cases, this part **requires proficiency with but not a deep understanding of**:

1. ``threading`` library of python
2. ``multiprocessing`` library of python
3. ``torch.distributed`` module
4. ``torch.distributed.rpc`` module

If below examples are not enough for you, please refer to `tests <https://github.com/iffiX/machin/tree/master/test>`_

Multi-threading examples
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
`Waiting on multiple events <https://github.com/iffiX/machin/blob/master/examples/tutorials/parallel_distributed/mth_event.py>`_

`Detect exception thrown in a sub-thread <https://github.com/iffiX/machin/blob/master/examples/tutorials/parallel_distributed/mth_exception.py>`_

`Using thread pools and context thread pools <https://github.com/iffiX/machin/blob/master/test/parallel/test_pool.py>`_

Multi-processing examples
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
`Serialization <https://github.com/iffiX/machin/blob/master/examples/tutorials/parallel_distributed/mpr_pickle.py>`_

`Detect exception thrown in a sub-process <https://github.com/iffiX/machin/blob/master/examples/tutorials/parallel_distributed/mpr_exception.py>`_

`Using pools and context pools <https://github.com/iffiX/machin/blob/master/test/parallel/test_pool.py>`_

Distributed examples
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
`Distributed world and collective group <https://github.com/iffiX/machin/blob/master/examples/tutorials/parallel_distributed/mpr_coll.py>`_

Distributed RPC examples
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
`Distributed world and rpc group <https://github.com/iffiX/machin/blob/master/examples/tutorials/parallel_distributed/dist_rpc.py>`_

`A simple key-value server with version tracking <https://github.com/iffiX/machin/blob/master/examples/tutorials/parallel_distributed/dist_oserver.py>`_

Model parallel examples
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
`Assigning models automatically <https://github.com/iffiX/machin/blob/master/examples/tutorials/parallel_distributed/assign.py>`_
