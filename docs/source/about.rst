About
==================================
.. toctree::
   :maxdepth: 1
   :caption: Contents:

Machin is a reinforcement library purely based on pytorch,
it is designed with three things in mind:

1. **Easy to understand.**
2. **Easy to extend.**
3. **Easy to reuse.**

The first goal is achieved through clear structure design, robust document,
and concise description of use cases. The second goal is achieved through
adding an extra layer upon basic apis provided in the distributed module of
pytorch, this layer offers additional fault tolerance mechanism and
eliminates hassles occurring in distributed programming. The last goal is
the result of modular designs, careful api arrangements, and experiences
gathered from other similar projects.

Compared to other versatile and powerful reinforcement learning frameworks,
Machin tries to offer a pleasant programming experience, smoothing out
as many obstacles involved in reinforcement learning and distributed
programming as possible. Some essential functions such as automated tuning and
neural architecture search are not offered in this package, we strongly
recommend you take a look at these amazing projects and take a piggyback ride:

* `ray tune <https://github.com/ray-project/ray/tree/master/python/ray/tune>`_
* `tpot <https://github.com/EpistasisLab/tpot>`_
* `nni <https://github.com/microsoft/nni>`_