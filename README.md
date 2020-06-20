<div align="center">
	<a href="https://machin.readthedocs.io">
		<img width="auto" height="200px" src="https://machin.readthedocs.io/en/latest/_static/icon.svg">
	</a>
</div>

<br/><br/>

[![Read the Docs](https://img.shields.io/readthedocs/machin)](https://machin.readthedocs.io/en/latest/)
[![Jenkins build](https://img.shields.io/jenkins/build?jobUrl=http%3A%2F%2Fci.beyond-infinity.com%2Fjenkins%2Fjob%2Fmachin%2Fjob%2Fmaster%2F)](http://ci.beyond-infinity.com/jenkins/blue/organizations/jenkins/machin/branches/)
[![Jenkins tests](https://img.shields.io/jenkins/tests?compact_message&jobUrl=http%3A%2F%2Fci.beyond-infinity.com%2Fjenkins%2Fjob%2Fmachin%2Fjob%2Fmaster%2F)](http://ci.beyond-infinity.com/jenkins/blue/organizations/jenkins/machin/branches/)
[![Jenkins coverage](https://img.shields.io/jenkins/coverage/cobertura?jobUrl=http%3A%2F%2Fci.beyond-infinity.com%2Fjenkins%2Fjob%2Fmachin%2Fjob%2Fmaster%2F)](http://ci.beyond-infinity.com/jenkins/blue/organizations/jenkins/machin/branches/)
[![License](https://img.shields.io/github/license/iffiX/machin)](https://github.com/iffiX/machin)

<br/>
WARNING: currently Machin is in its early development stage, tests, code coverage, doc, examples are not complete.


<br/>
**Machin** is a reinforcement library purely based on pytorch. It is designed to be **readable**, **reusable** and **extendable**.


### Supported algorithms
---
Currently Machin has implemented the following algorithms, the list is still growing:

#### Single agent algorithms:
* [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
* [Dueling DQN](https://arxiv.org/abs/1511.06581)
* [RAINBOW](https://arxiv.org/abs/1710.02298)
* [Deep Deterministic policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
* [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
* [Hystereric DDPG (Modified from Hys-DQN)](https://hal.archives-ouvertes.fr/hal-00187279/document)
* [Advantage Actor-Critic (A2C)](https://openai.com/blog/baselines-acktr-a2c/)
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
* [Soft Actor Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)

#### Multi-agent algorithms:
* [Multi-agent DDPG (MADDPG)](https://arxiv.org/pdf/1706.02275.pdf)

#### Massively parallel algorithms:
* [Asynchronous A2C (A3C)](https://arxiv.org/abs/1602.01783)
* [APEX-DQN, APEX-DDPG](https://arxiv.org/pdf/1803.00933)
* [IMPALA](https://arxiv.org/pdf/1802.01561)

#### Enhancements:
* [Prioritized Experience Replay (PER)](https://arxiv.org/pdf/1511.05952.pdf)
* [Generalized Advantage Estimation (GAE)](https://arxiv.org/pdf/1506.02438.pdf)
* [Recurrent networks in DQN, etc.](https://arxiv.org/pdf/1507.06527.pdf)
#### Algorithms to be supported:
* [Distributed DDPG (D4PG)](https://arxiv.org/abs/1804.08617)
* [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/abs/1606.03476)
* Evolution Strategies
* [QMIX (multi agent)](https://arxiv.org/abs/1803.11485)
* Model-based methods

### Features
---
#### 1. Readable

The initial development drive for the Machin library originates from one common plague of reinforcement learning libraries such as the famous [rlpyt](https://github.com/astooke/rlpyt), [ray](https://github.com/ray-project/ray), and [baselines](https://github.com/openai/baselines): **complexity**. Machin tries to just provide a simple, clear implementation of RL algorithms and bring as little obstacles to users as possible.

Therefore, Machin is designed with minimial abstractions and comes with a very detailed document, as well as various tutorials to help you build your own program.

#### 2. Reusable

Machin takes a similar approach to that of pytorch, encasulating algorithms, data structures in their own classes. Users do not need to setup a series of `data collectors`, `trainers`, `runners`, `samplers`... to use them, **just import**. Although Machin do provide some parallel wrappers to aid you construct high-performance programs, **you are still the operator of your dataflow**.

The only restriction placed on your models is their input / output format, however, these restrictions are minimal, and can still be customized to make Machin work happily. 

#### 3. Extendable
Machin is built upon pytorch, and thanks to its powerful rpc api, we may construct complex distributed programs, Machin provides a reliable rpc execution layer upon raw rpc apis, based on a stable election algorithm, so that you can ignore threads, ranks, processes, and just need to define services! A service is much like a http server which can be moved around freely, will **never fail** (from clients' perspective) as long as you define how to restart after a crash. 

For production grade services, you might will want to have some persistency and consistency, but those are not what Machin designed to do, you will have to do it yourself, by using databases, log systems, etc..

Apart from rpc, Machin also provides enhancements on multiprocessing pools so that you may use **lambdas and local functions**, and a light encapsule layer of collective communications (MPI style).

#### 4. Reproducible
Machin is reproducible, because for each release, our test framework will directly train every RL framework, if any framework cannot reach the target score, the test will fail directly.


### Documentation
---
See [here](https://machin.readthedocs.io/). Examples are located in [examples](https://github.com/iffiX/machin/tree/master/examples).

### Installation
---
Machin is hosted on [PyPI](https://pypi.org/project/machin/). Python >= 3.5 and PyTorch >= 1.5.0 is required. You may install the Machin library by simply typing:
```
pip3 install machin
```

### Contributing
---
Any contribution would be great, don't hesitate to submit a PR request to us! Please follow the instructions in [this](https://github.com/iffiX/machin/tree/master/docs/misc/contribute.md) file.

### Issues
---
If you have any issues, please use the template markdown files in [.github/ISSUE_TEMPLATE](https://github.com/iffiX/machin/tree/master/.github/ISSUE_TEMPLATE) 
folder and  format your issue before opening a new one. We would try our best to respond to your feature requests and problems.

### Citing
---
We would be very grateful if you can cite our work in your publications:
```
@misc{machin,
  author = {Muhan Li},
  title = {Machin},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/iffiX/machin}},
}
```

### Roadmap
---
Please see [Roadmap](https://github.com/iffiX/machin/projects/2) for the exciting new features we are currently working on!
