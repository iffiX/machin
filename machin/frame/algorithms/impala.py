from typing import Union, Dict, List, Tuple, Callable, Any
from copy import deepcopy
import random
import numpy as np
import torch as t
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from machin.frame.buffers.buffer_d import DistributedBuffer
from machin.frame.transition import Transition
from machin.frame.helpers.servers import model_server_helper
from machin.model.nets.base import NeuralNetworkModule
from machin.parallel.server import PushPullModelServer
from machin.parallel.distributed import RpcGroup, get_world
from .base import TorchFramework, Config
from .utils import (
    safe_call,
    assert_and_get_valid_models,
    assert_and_get_valid_optimizer,
    assert_and_get_valid_criterion,
    assert_and_get_valid_lr_scheduler,
)


def _disable_update(*_, **__):
    return None, None


class IMPALABuffer(DistributedBuffer):
    """
    Samples full episodes for batch_size instead of steps.
    """

    def sample_batch(
        self,
        batch_size: int,
        concatenate: bool = True,
        device: Union[str, t.device] = "cpu",
        sample_attrs: List[str] = None,
        additional_concat_custom_attrs: List[str] = None,
        *_,
        **__,
    ) -> Any:
        return super().sample_batch(
            batch_size=batch_size,
            concatenate=concatenate,
            device=device,
            sample_method="episode",
            sample_attrs=sample_attrs,
            additional_concat_custom_attrs=additional_concat_custom_attrs,
        )

    def sample_method_episode(self, batch_size: int) -> Tuple[int, List[Transition]]:
        """
        Args:
            batch_size: Number of **episodes** to sample.
        """
        batch_size = min(len(self.episode_transition_handles), batch_size)
        episodes = random.choices(
            list(self.episode_transition_handles.keys()), k=batch_size
        )
        batch = [
            self.storage[handle]
            for episode in episodes
            for handle in self.episode_transition_handles[episode]
        ]
        return batch_size, batch


class IMPALA(TorchFramework):
    """
    Massively parallel IMPALA framework.
    """

    _is_top = ["actor", "critic"]
    _is_restorable = ["actor", "critic"]

    def __init__(
        self,
        actor: Union[NeuralNetworkModule, nn.Module],
        critic: Union[NeuralNetworkModule, nn.Module],
        optimizer: Callable,
        criterion: Callable,
        impala_group: RpcGroup,
        model_server: Tuple[PushPullModelServer],
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple, Tuple] = (),
        lr_scheduler_kwargs: Tuple[Dict, Dict] = (),
        batch_size: int = 5,
        learning_rate: float = 0.001,
        isw_clip_c: float = 1.0,
        isw_clip_rho: float = 1.0,
        entropy_weight: float = None,
        value_weight: float = 0.5,
        gradient_max: float = np.inf,
        discount: float = 0.99,
        replay_size: int = 500,
        **__,
    ):
        """
        Note:
            Please make sure isw_clip_rho >= isw_clip_c
        Args:
            actor: Actor network module.
            critic: Critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            impala_group: Group of all processes using the IMPALA framework,
                including all samplers and trainers.
            model_server: Custom model sync server accessor for ``actor``.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            batch_size: Batch size used during training.
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            isw_clip_c: :math:`c` used in importance weight clipping.
            isw_clip_rho:
            entropy_weight: Weight of entropy in your loss function, a positive
                entropy weight will minimize entropy, while a negative one will
                maximize entropy.
            value_weight: Weight of critic value loss.
            gradient_max: Maximum gradient.
            discount: :math:`\\gamma` used in the bellman function.
            replay_size: Size of the local replay buffer.
        """
        self.batch_size = batch_size
        self.discount = discount
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.grad_max = gradient_max
        self.isw_clip_c = isw_clip_c
        self.isw_clip_rho = isw_clip_rho

        self.impala_group = impala_group

        self.actor = actor
        self.critic = critic
        self.actor_optim = optimizer(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = optimizer(self.critic.parameters(), lr=learning_rate)
        self.replay_buffer = IMPALABuffer(
            buffer_name="buffer", group=impala_group, buffer_size=replay_size
        )
        self.is_syncing = True
        self.actor_model_server = model_server[0]

        if lr_scheduler is not None:
            self.actor_lr_sch = lr_scheduler(
                self.actor_optim, *lr_scheduler_args[0], **lr_scheduler_kwargs[0],
            )
            self.critic_lr_sch = lr_scheduler(
                self.critic_optim, *lr_scheduler_args[1], **lr_scheduler_kwargs[1]
            )

        self.criterion = criterion
        self._is_using_DP_or_DDP = isinstance(
            self.actor, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
        )
        super().__init__()

    @property
    def optimizers(self):
        return [self.actor_optim, self.critic_optim]

    @optimizers.setter
    def optimizers(self, optimizers):
        self.actor_optim, self.critic_optim = optimizers

    @property
    def lr_schedulers(self):
        if hasattr(self, "actor_lr_sch") and hasattr(self, "critic_lr_sch"):
            return [self.actor_lr_sch, self.critic_lr_sch]
        return []

    @classmethod
    def is_distributed(cls):
        return True

    def set_sync(self, is_syncing):
        self.is_syncing = is_syncing

    def manual_sync(self):
        if not self._is_using_DP_or_DDP:
            self.actor_model_server.pull(self.actor)

    def act(self, state: Dict[str, Any], *_, **__):
        """
        Use actor network to give a policy to the current state.

        Returns:
            Anything produced by actor.
        """
        if self.is_syncing and not self._is_using_DP_or_DDP:
            self.actor_model_server.pull(self.actor)
        return safe_call(self.actor, state)

    def _eval_act(self, state: Dict[str, Any], action: Dict[str, Any], *_, **__):
        """
        Use actor network to evaluate the log-likelihood of a given
        action in the current state.

        Returns:
            Anything produced by actor.
        """
        return safe_call(self.actor, state, action)

    def _criticize(self, state: Dict[str, Any], *_, **__):
        """
        Use critic network to evaluate current value.

        Returns:
            Value of shape ``[batch_size, 1]``
        """
        return safe_call(self.critic, state)[0]

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.
        """
        if len(episode) == 0:
            raise ValueError("Episode must be non-empty.")

        # The first step records episode length, other steps records 0
        episode[0]["episode_length"] = len(episode)
        for transition in episode[1:]:
            transition["episode_length"] = 0

        self.replay_buffer.store_episode(
            episode,
            required_attrs=(
                "state",
                "action",
                "next_state",
                "reward",
                "action_log_prob",
                "terminal",
                "episode_length",
            ),
        )

    def update(self, update_value=True, update_policy=True, **__):
        """
        Update network weights by sampling from replay buffer.

        Note:
            Will always concatenate samples.

        Args:
            update_value: Whether to update the Q network.
            update_policy: Whether to update the actor network.

        Returns:
            mean value of estimated policy value, value loss
        """
        # sample a batch

        # `batch_size` here means the number of episodes sampled, not
        # the number of steps sampled.
        # The size of the batch dimension of sampled attributes should be
        # the summed length of sampled episodes,
        # eg: total_length = ep1_length + ep2_length + ...

        self.actor.train()
        self.critic.train()
        (
            batch_size,
            (
                state,
                action,
                reward,
                next_state,
                terminal,
                action_log_prob,
                episode_length,
            ),
        ) = self.replay_buffer.sample_batch(
            self.batch_size,
            device="cpu",
            sample_attrs=[
                "state",
                "action",
                "reward",
                "next_state",
                "terminal",
                "action_log_prob",
                "episode_length",
            ],
            additional_concat_custom_attrs=["action_log_prob"],
        )

        # episodes are chained together like:
        # ep1_step1, ep1_step2, ..., ep1_stepN, ep2_step1, ep2_step2 ...

        # `state`, `action` and `next_state` should be dicts like:
        # {"attr1": Tensor(total_length, ...),
        #  "attr2": Tensor(total_length, ...)}

        # `terminal`, `reward`, `action_log_prob` should be tensors like:
        # Tensor(total_length, 1)

        # store the length of each episode, so that we can find boundaries
        # between two episodes inside the chained "sample"
        all_length = [length for length in episode_length if length != 0]
        sum_length = sum(all_length)

        if sum_length != terminal.shape[0]:
            raise RuntimeError(
                "Sum length is unequal to tensor total length,"
                " an unknown error has occurred."
            )

        # Below are the v-trace process

        # Calculate c and rho first, because there is no dependency
        # between vector elements.
        _, cur_action_log_prob, entropy, *__ = self._eval_act(state, action)
        cur_action_log_prob = cur_action_log_prob.view(sum_length, 1).to("cpu")
        entropy = entropy.view(sum_length, 1).to("cpu")

        # similarity = pi(a_t|x_t)/mu(a_t|x_t)
        sim = t.exp(cur_action_log_prob - action_log_prob)
        c = t.min(t.full(sim.shape, self.isw_clip_c, dtype=sim.dtype), sim)
        rho = t.min(t.full(sim.shape, self.isw_clip_rho, dtype=sim.dtype), sim)

        # calculate delta V
        # delta_t V = rho_t(r_t + gamma * V(x_{t+1}) - V(x_t))
        # boundary elements (i.e, ep1_stepN) will have V(x_{t+1}) = 0
        value = self._criticize(state).view(sum_length, 1).to("cpu")
        next_value = self._criticize(next_state).view(sum_length, 1).to("cpu")
        next_value[terminal] = 0
        delta_v = rho * (reward + self.discount * next_value - value)

        # calculate v_s

        # vss is v_s shifted left by 1 element, i.e. becomes v_{s+1}
        # boundary elements (i.e, ep1_stepN) will have v_{s+1} = 0

        # do reversed cumulative product for each episode segment
        with t.no_grad():
            vs = t.zeros_like(value)
            vss = t.zeros_like(value)
            offset = 0
            for ep_len in all_length:
                # the last v_s of each episode should be 0
                # or V_t - rho_t * (r_t - V_t)? (since Vt+1 = 0)
                # Implementations such as
                # https://github.com/junjungoal/IMPALA-pytorch/blob/master
                # /agents/learner.py   use the first case, 0
                # 0 works well when rho=c=1 or 1 > rho >= c
                for rev_step in reversed(range(ep_len - 1)):
                    idx = offset + rev_step
                    vs[idx] = (
                        value[idx]
                        + delta_v[idx]
                        + self.discount * c[idx] * (vs[idx + 1] - value[idx + 1])
                    )
                # shift v_s to get v_{s+1}
                vss[offset : offset + ep_len - 1] = vs[offset + 1 : offset + ep_len]

                # update offset
                offset += ep_len

        act_policy_loss = -(
            rho.detach()
            * cur_action_log_prob
            * (reward + self.discount * vss - value).detach()
        )
        if self.entropy_weight is not None:
            act_policy_loss += self.entropy_weight * entropy
        act_policy_loss = act_policy_loss.sum()

        value_loss = self.criterion(value, vs.type_as(value))

        # Update actor network
        if update_policy:
            self.actor.zero_grad()
            self._backward(act_policy_loss)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_max)
            self.actor_optim.step()

        # Update critic network
        if update_value:
            self.critic.zero_grad()
            self._backward(value_loss)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_max)
            self.critic_optim.step()

        # push actor model for samplers
        if self._is_using_DP_or_DDP:
            self.actor_model_server.push(self.actor.module, pull_on_fail=False)
        else:
            self.actor_model_server.push(self.actor)

        self.actor.eval()
        self.critic.eval()
        return -act_policy_loss.item(), value_loss.item()

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "actor_lr_sch"):
            self.actor_lr_sch.step()
        if hasattr(self, "critic_lr_sch"):
            self.critic_lr_sch.step()

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        default_values = {
            "learner_process_number": 1,
            "model_server_group_name": "impala_model_server",
            "model_server_members": "all",
            "impala_group_name": "impala",
            "impala_members": "all",
            "models": ["Actor", "Critic"],
            "model_args": ((), ()),
            "model_kwargs": ({}, {}),
            "optimizer": "Adam",
            "criterion": "MSELoss",
            "criterion_args": (),
            "criterion_kwargs": {},
            "lr_scheduler": None,
            "lr_scheduler_args": None,
            "lr_scheduler_kwargs": None,
            "batch_size": 5,
            "learning_rate": 0.001,
            "isw_clip_c": 1.0,
            "isw_clip_rho": 1.0,
            "entropy_weight": None,
            "value_weight": 0.5,
            "gradient_max": np.inf,
            "discount": 0.99,
            "replay_size": 500,
        }
        config = deepcopy(config)
        config["frame"] = "IMPALA"
        config["batch_num"] = {"sampler": 10, "learner": 1}
        if "frame_config" not in config:
            config["frame_config"] = default_values
        else:
            config["frame_config"] = {**config["frame_config"], **default_values}
        return config

    @classmethod
    def init_from_config(
        cls,
        config: Union[Dict[str, Any], Config],
        model_device: Union[str, t.device] = "cpu",
    ):
        world = get_world()
        f_config = deepcopy(config["frame_config"])
        impala_group = world.create_rpc_group(
            group_name=f_config["impala_group_name"],
            members=(
                world.get_members()
                if f_config["impala_members"] == "all"
                else f_config["impala_members"]
            ),
        )

        models = assert_and_get_valid_models(f_config["models"])
        model_args = f_config["model_args"]
        model_kwargs = f_config["model_kwargs"]
        models = [
            m(*arg, **kwarg).to(model_device)
            for m, arg, kwarg in zip(models, model_args, model_kwargs)
        ]
        # wrap models in DistributedDataParallel when running in learner mode
        max_learner_id = f_config["learner_process_number"]

        learner_group = world.create_collective_group(ranks=list(range(max_learner_id)))

        if world.rank < max_learner_id:
            models = [
                DistributedDataParallel(module=m, process_group=learner_group.group)
                for m in models
            ]

        optimizer = assert_and_get_valid_optimizer(f_config["optimizer"])
        criterion = assert_and_get_valid_criterion(f_config["criterion"])(
            *f_config["criterion_args"], **f_config["criterion_kwargs"]
        )
        lr_scheduler = f_config["lr_scheduler"] and assert_and_get_valid_lr_scheduler(
            f_config["lr_scheduler"]
        )
        servers = model_server_helper(
            model_num=1,
            group_name=f_config["model_server_group_name"],
            members=f_config["model_server_members"],
        )
        del f_config["optimizer"]
        del f_config["criterion"]
        del f_config["lr_scheduler"]
        frame = cls(
            *models,
            optimizer,
            criterion,
            impala_group,
            servers,
            lr_scheduler=lr_scheduler,
            **f_config,
        )
        if world.rank >= max_learner_id:
            frame.role = "sampler"
            frame.update = _disable_update
        else:
            frame.role = "learner"
        return frame
