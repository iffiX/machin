As fast as lightning
=======================
**Author**: `Muhan Li <https://github.com/iffiX>`_

Its a pain to lay down every details by hand, so why don't we do it automatically?
We have `PyTorch Lightning <https://www.pytorchlightning.ai/>`_, which is a powerful
machine learning programming utility that enables users to write template like code
and leave details such as check-pointing, logging, metric evaluation to the Lightning
engine hidden behind. The Machin library also supports coding with the Lightning library
to simplify your RL workflow.

The automatic launcher
--------------------------------
**Full code**: `Github <https://github.com/iffiX/machin/blob/master/examples/tutorials/as_fast_as_lightning/automatic/>`_


The simplest way to use PyTorch Lightning is using the automatic launcher.

Generating and modifying the config
++++++++++++++++++++++++++++++++
First you need to generate a complete config, including the framework config and the environment config::

    python -m machin.auto generate --algo DQN --env openai_gym --output config.json

The `config.json` looks like this::

    {
        "early_stopping_patience": 3,
        "env": "openai_gym",
        "episode_per_epoch": 10,
        "frame": "DQN",
        "frame_config": {
            "batch_size": 100,
            "criterion": "MSELoss",
            "criterion_args": [],
            "criterion_kwargs": {},
            "discount": 0.99,
            "epsilon_decay": 0.9999,
            "gradient_max": Infinity,
            "learning_rate": 0.001,
            "lr_scheduler": null,
            "lr_scheduler_args": null,
            "lr_scheduler_kwargs": null,
            "mode": "double",
            "model_args": [
                [],
                []
            ],
            "model_kwargs": [
                {},
                {}
            ],
            "models": [
                "QNet",
                "QNet"
            ],
            "optimizer": "Adam",
            "replay_buffer": null,
            "replay_device": "cpu",
            "replay_size": 500000,
            "update_rate": 0.005,
            "update_steps": null,
            "visualize": false,
            "visualize_dir": ""
        },
        "gpus": [
            0
        ],
        "max_episodes": 10000,
        "root_dir": "trial",
        "test_env_config": {
            "act_kwargs": {},
            "env_name": "CartPole-v1",
            "render_every_episode": 100
        },
        "train_env_config": {
            "act_kwargs": {},
            "env_name": "CartPole-v1",
            "render_every_episode": 100
        }
    }

then you need to modify this config file to suit your needs, first we need to modify
the framework config stored under sub-key `"frame_config"`:

    1. You need to define your `QNet` model in some file, suppose you defined it in `qnet.py`,
       then in the same directory, set key `"models"` to `["qnet.QNet", "qnet.QNet"]`, currently
       this is the only way when you use Lightning with the automatic launcher. There are other
       ways which are described below.
    2. If your model has any initialization `args` and `kwargs`, then you will also need to
       specify them for each one of your model.
    3. Other keys corresponds to the initialization argument of :class:`.DQN`. Please refer
       to its docstring for more information.

After modifying the framework config, you also need to modify the environemt config when you need to. The
environment config is provided to pytorch datasets which wraps the target environment defined in `machin.auto.envs`.
For example, the `openai_gym` module defines two dataset classes for discrete and continuous actions:
:class:`.RLGymDiscActDataset` and :class `.RLGymContActDataset`.

For the OpenAI Gym environemt, `"act_kwargs"` is the keyword arguments passed to `act` or `act_*_with_noise`
depending on your framework, please refer to :class:`.RLGymDiscActDataset` and :class `.RLGymContActDataset`
for their specific usage.

Finally, you may also want to modify other configuration keys passed to the Lightning framework:

    1. `"early_stopping_patience"`: the maximum number of epochs where total reward does not increase before
       terminating training, this value is passed to the `EarlyStopping` hook in Lightning.
    2. `"episode_per_epoch"`: Number of episodes to run per epoch.
    3. `"max_episodes"`: Number of maximum training episodes.
    4. `"root_dir"`: Root directory to use for check-pointing, logging, etc. in this training. Must be unique
       for each experiment, otherwise your results will be overwritten.
    5. `"gpus"`: number of gpus to train on (int) or which GPUs to train on (list or str) applied per node,
       passed to `pytorch_lightning.Trainer`.

For distributed frameworks such as IMPALA, there are two additional configuration keys:

    1. `"num_nodes"`: number of nodes (int) for distributed training,
       passed to `pytorch_lightning.Trainer`.
    2. `"num_processes"`: number of processes to run (int) on each node for distributed training,
       only necessary when you set key `"gpu"` to `null` indicating that you will not use gpu but
       cpu only, otherwise the created process number is equal to the gpu number specified.

Launching the experiment
++++++++++++++++++++++++++++++++
You can simply call::

    python -m machin.auto launch --config config.json

To launch your experiment with pytorch lightning.

Limitation
++++++++++++++++++++++++++++++++
The main limitation is that: you cannot use any custom environment except already provided ones. And it is also
inflexible for hyper parameter searching when you want to fine-tune your model. Therefore, we will first introduce
you to use the `auto` module programmatically instead of declaratively, then instruct you to create your own
environment extension.


Programming with the auto module
--------------------------------
**Full code**: `Github <https://github.com/iffiX/machin/blob/master/examples/tutorials/as_fast_as_lightning/programmatic/>`_

Suppose that you want to sweep the hyper parameter space using some tuning library like
`Microsoft NNI <https://github.com/microsoft/nni>`_, then you program can be divided into
the following pseudo program::

    Init NNI with some hyper parameter space.
    While NNI has next parameter p:
        start experiment with parameter p
        report NNI with performance

First step: generate framework config
++++++++++++++++++++++++++++++++
You can generate a dictionary-like framework config by invoking the :func`.generate_algorithm_config`::

    from machin.auto.config import (
        generate_algorithm_config,
        generate_env_config,
        generate_training_config,
        launch,
    )

    config = generate_algorithm_config("DQN")

You will get some result like::

    Out[1]:
    {'frame': 'DQN',
     'frame_config': {'models': ['QNet', 'QNet'],
      'model_args': ((), ()),
      'model_kwargs': ({}, {}),
      'optimizer': 'Adam',
      'criterion': 'MSELoss',
      'criterion_args': (),
      'criterion_kwargs': {},
      'lr_scheduler': None,
      'lr_scheduler_args': None,
      'lr_scheduler_kwargs': None,
      'batch_size': 100,
      'epsilon_decay': 0.9999,
      'update_rate': 0.005,
      'update_steps': None,
      'learning_rate': 0.001,
      'discount': 0.99,
      'gradient_max': inf,
      'replay_size': 500000,
      'replay_device': 'cpu',
      'replay_buffer': None,
      'mode': 'double',
      'visualize': False,
      'visualize_dir': ''},
     'gpus': [0]}

.. Warning::
    This method generated some PyTorch Lightning specific configurations, which
    are not provided by the `generate_config` method in algorithms classes,
    the `generate_config` class method is designed to only initialize the algorithm
    framework with the `init_from_config` class method of that algorithm class.

All framework related configurations are under the sub-key "frame_config".

.. Note::
    You can pass the model class defined by you to the framework in the following
    ways:

    1. An `nn.Module` subclass
    2. A string name of a global defined model class in any frame
       of your call stack. (Not available if framework is distributed),
    3. A string name of an importable model class, eg: foo.baz.model

    example::

        class QNet(nn.Module):
            ...

        # specify directly
        config["frame_config"]["models"] = [QNet, QNet]

        # specify as a string, since it is defined globally in the current stack
        config["frame_config"]["models"] = ["QNet", "QNet"]

        # specify as a importable name, current module is "__main__"
        config["frame_config"]["models"] = ["__main__.QNet", "__main__.QNet"]


.. Note::
    For `optimizer`, `lr_scheduler`, `criterion`, etc. you can specify them in the same way you
    specify your models, they have an additional way to define: a valid string name of
    some respective class in the PyTorch library, please refer to
    :func:`.assert_and_get_valid_optimizer` and :func:`.assert_and_get_valid_lr_scheduler`
    and :func:`.assert_and_get_valid_criterion`.

You can fill in your hyper parameters provided by NNI into the `"frame_config"` section.

Second step: generate environment config
++++++++++++++++++++++++++++++++
This step is straightforward, select your target environment and generate::

    config = generate_env_config("openai_gym", config)

Which will add the following keys::

    "test_env_config": {
        "act_kwargs": {},
        "env_name": "CartPole-v1",
        "render_every_episode": 100
    },
    "train_env_config": {
        "act_kwargs": {},
        "env_name": "CartPole-v1",
        "render_every_episode": 100
    }

Third step: generate training config
++++++++++++++++++++++++++++++++
Training config are provided to PyTorch Lightning `Trainer` class::

    config = generate_training_config(root_dir=".trial",
                                      episode_per_epoch=10,
                                      max_episodes=10000,
                                      config=config)

Final step: launch with generated config
++++++++++++++++++++++++++++++++
Just pass your config dictionary to the launcher:

    launch(config)


Writing an environment extension
--------------------------------
**Code reference**: `Github <https://github.com/iffiX/machin/blob/master/machin/auto/envs/openai_gym.py>`_

All environment adaptors are located in `machin.auto.envs`, to create an environment extension,
you need to:

    1. Create a python file with your environment name, such as "some_env.py".
    2. Update `__init__.py` in `machin.auto.envs` to import your environment
       module as a whole, this is used to look up available environments.

For your environment module, you need to define 4 things:

    1. A dataset class which inherits and implements methods defined in
       :class:`.RLDataset`, when `__next__` method is called, it must return
       a sampled episode of type :class:`DatasetResult`.
    2. A dataset creator function which takes in a framework instance (such as
       an instance of DQN) and pass this to the dataset so the framework can be
       used internally to interact with your environment. It must return a dataset
       class instance.
    3. A function named `generate_env_config` which takes in a previous config
       and add three keys: `"env"`, `"train_env_config"`, and `"test_env_config"`,
       `"env"` is your environment name, and two configs are used to initialize
       the test and train environment.
    4. A launch function which takes in a config object and a list of PyTorch Lightning
       callbacks, it is used to launch the experiment with PyTorch Lightning `Trainer`.
