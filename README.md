<div align="center">
	<a href="https://github.com/iffiX/machin">
		<img width="auto" height="200px" src="https://machin.readthedocs.io/en/latest/_static/icon.png">
	</a>
</div>

<br/>


<div align="center">
	<a href="https://machin.readthedocs.io/en/latest/">
		<img alt="Read the Docs" src="https://img.shields.io/readthedocs/machin">
	</a>
	<a href="http://ci.beyond-infinity.com/jenkins/blue/organizations/jenkins/machin/branches/">
		<img alt="Jenkins build" src="https://img.shields.io/jenkins/build?jobUrl=http%3A%2F%2Fci.beyond-infinity.com%2Fjenkins%2Fjob%2Fmachin%2Fjob%2Fmaster%2F">
	</a>
	<a href="http://ci.beyond-infinity.com/jenkins/blue/organizations/jenkins/machin/branches/">
		<img alt="Jenkins coverage" src="https://img.shields.io/jenkins/coverage/cobertura?jobUrl=http%3A%2F%2Fci.beyond-infinity.com%2Fjenkins%2Fjob%2Fmachin%2Fjob%2Frelease%2F">
	</a>
	<a href="https://pypi.org/project/machin/">
		<img alt="PyPI version" src="https://img.shields.io/pypi/v/machin">
	</a>
	<a href="https://github.com/iffiX/machin">
		<img alt="License" src="https://img.shields.io/github/license/iffiX/machin">
	</a>
	<a href="http://ci.beyond-infinity.com/reports/machin/">
		<img alt="Report" src="https://img.shields.io/badge/report-allure-blue">
	</a>
	<a href="https://github.com/psf/black">
		<img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
	</a>
</div>

<br/>

<div align="center">
<p><strong>Automatic, Readable, Reusable, Extendable</strong></p>

<p><strong>Machin</strong> is a reinforcement library designed for pytorch.</p> 
</div>

<br/>

### Supported Models
---
**Anything**, including recurrent networks.

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
* [Augmented random search (ARS, non-gradient)](https://arxiv.org/pdf/1803.07055.pdf)

#### Enhancements:
* [Prioritized Experience Replay (PER)](https://arxiv.org/pdf/1511.05952.pdf)
* [Generalized Advantage Estimation (GAE)](https://arxiv.org/pdf/1506.02438.pdf)
* [Recurrent networks in DQN, etc.](https://arxiv.org/pdf/1507.06527.pdf)
#### Algorithms to be supported:
* [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/abs/1606.03476)
* Evolution Strategies
* [QMIX (multi agent)](https://arxiv.org/abs/1803.11485)
* Model-based methods

### Features
---

#### 1. Automatic

Starting from version 0.4.0, Machin now supports automatic config generation, you can get a configuration
through:
```
python -m machin.auto generate --algo DQN --env openai_gym --output config.json
```

And automatically launch the experiment with pytorch lightning:
```
python -m machin.auto launch --config config.json
```


#### 2. Readable

Compared to other reinforcement learning libraries such as the famous [rlpyt](https://github.com/astooke/rlpyt), [ray](https://github.com/ray-project/ray), and [baselines](https://github.com/openai/baselines). Machin tries to just provide a simple, clear implementation of RL algorithms.

All algorithms in Machin are designed with minimial abstractions and have very detailed documents, as well as various helpful tutorials.

#### 3. Reusable

Machin takes a similar approach to that of pytorch, encasulating algorithms, data structures in their own classes. Users do not need to setup a series of `data collectors`, `trainers`, `runners`, `samplers`... to use them, **just import**.

The only restriction placed on your models is their input / output format, however, these restrictions are minimal, making it easy to adapt algorithms to your custom environments. 

#### 4. Extendable
Machin is built upon pytorch, it and thanks to its powerful rpc api, we may construct complex distributed programs. Machin provides implementations for enhanced parallel execution pools, automatic model assignment, role based rpc scaling, rpc service discovery and registration, etc.

Upon these core functions, Machin is able to provide tested high-performance distributed training algorithm implementations, such as A3C, APEX, IMPALA, to ease your design.

#### 5. Reproducible
Machin is **weakly** reproducible, for each release, our test framework will directly train every RL framework, if any framework cannot reach the target score, the test will fail directly. 

However, currently, the tests are not guaranteed to
be exactly the same as the tests in original papers, due to the large variety of different environments used in original research papers.


### Documentation
---
See [here](https://machin.readthedocs.io/). Examples are located in [examples](https://github.com/iffiX/machin/tree/master/examples).

### Installation
---
Machin is hosted on [PyPI](https://pypi.org/project/machin/). Python >= 3.6 and PyTorch >= 1.6.0 is required. You may install the Machin library by simply typing:
```
pip install machin
```
You are suggested to create a virtual environment first if you are using conda to manage your environments, to prevent PIP changes your packages without letting
conda know.
```
conda create -n some_env pip
conda activate some_env
pip install machin
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
