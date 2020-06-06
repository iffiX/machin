import copy
import itertools
import numpy as np
from machin.frame.transition import TransitionBase
from machin.utils.logging import default_logger
from machin.parallel.pool import Pool
from machin.parallel.assigner import ModelAssigner
# pylint: disable=wildcard-import, unused-wildcard-import
from .ddpg import *


def _average_parameters(*params: t.Tensor):
    # Average parameters from all sub policies, store the
    # averaged result in the first sub policy, then broadcast
    # result to remaining sub policies.
    target_param = params[0]
    target_param.data.copy_(
        t.mean(
            t.stack([p.to(target_param.device) for p in params], dim=0),
            dim=0
        )
    )
    for param in params[1:]:
        param.data.copy_(target_param)


class MultiAgentTransition(TransitionBase):
    """
    Multi agent Transition class used in MADDPG.
    """
    state = None  # type: Dict[str, t.Tensor]
    all_states = None  # type: Dict[str, t.Tensor]
    all_actions = None  # type: Dict[str, t.Tensor]
    all_next_states = None  # type: Dict[str, t.Tensor]
    reward = None  # type: Union[float, t.Tensor]
    terminal = None  # type: bool

    def __init__(self,
                 state: Dict[str, t.Tensor],
                 all_states: Dict[str, t.Tensor],
                 all_actions: Dict[str, t.Tensor],
                 all_next_states: Dict[str, t.Tensor],
                 reward: Union[float, t.Tensor],
                 index: int,
                 terminal: bool,
                 **kwargs):
        super(MultiAgentTransition, self).__init__(
            major_attr=["state", "all_states", "all_actions",
                        "all_next_states"],
            sub_attr=["reward", "terminal"],
            custom_attr=kwargs.keys(),
            major_data=[state, all_states, all_actions, all_next_states],
            sub_data=[reward, terminal, index],
            custom_data=kwargs.values()
        )


class MADDPG(TorchFramework):
    """
    MADDPG is a centralized multi-agent training framework, it alleviates the
    unstable reward problem caused by the disturbance of other agents by
    gathering all agents observations and train a global critic. This global
    critic observes all actions and all states from all agents.
    """
    # Since the number of sub-policies is automatically determined,
    # they are not considered here.
    _is_top = ["actor_target", "critic_target"]
    _is_restorable = ["actor_target", "critic_target"]

    def __init__(self,
                 agent_num,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 actor_target: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 critic_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer: Callable,
                 criterion: Callable,
                 available_devices: Union[list, None] = None,
                 sub_policy_num: int = 1,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple, Tuple, Tuple] = (),
                 lr_scheduler_kwargs: Tuple[Dict, Dict, Dict] = (),
                 batch_size: int = 100,
                 update_rate: float = 0.005,
                 learning_rate: float = 0.001,
                 discount: float = 0.99,
                 replay_size: int = 500000,
                 replay_device: Union[str, t.device] = "cpu",
                 replay_buffer: Buffer = None,
                 reward_func: Callable = None,
                 action_concat_func: Callable = None,
                 action_alter_func: Callable = None,
                 state_split_func: Callable = None,
                 visualize: bool = False,
                 pool: Any = None):
        """
        See Also:
            :class:`.DDPG`

        Hint:
            Please reference:

            - :meth:`MADDPG.action_concat_function`
            - :meth:`MADDPG.action_alter_function`
            - :meth:`MADDPG.state_split_function`

            for ``action_concat_func``, ``action_alter_func``, and
            ``state_split_func`` design, if your actor does not output
            a simple action of shape ``[batch_size, action_dim]`` which
            should be directly accepted by your critic.

        Args:
            actor: Actor network module.
            actor_target: Target actor network module.
            critic: Critic network module.
            critic_target: Target critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            batch_size: Batch size used during training.
            update_rate: :math:`\\tau` used to update target networks.
                Target parameters are updated as:
                :math:`\\theta_t = \\theta * \\tau + \\theta_t * (1 - \\tau)`
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            discount: :math:`\\gamma` used in the bellman function.
            replay_size: Replay buffer size. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer.
            reward_func: Reward function used in training.
            action_concat_func: Action concatenation function.
            action_alter_func: Action alternation function.
            state_split_func: All states spli function.
            visualize: Whether visualize the network flow in the first pass.
        """
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.visualize = visualize
        self.agent_num = agent_num

        self.actors = [copy.deepcopy(actor)
                       for _ in range(sub_policy_num)]
        self.actor_targets = [copy.deepcopy(actor_target)
                              for _ in range(sub_policy_num)]
        self.critics = [copy.deepcopy(critic)
                        for _ in range(sub_policy_num)]
        self.critic_targets = [copy.deepcopy(critic_target)
                               for _ in range(sub_policy_num)]
        self.actor_optims = [optimizer(ac.parameters(), lr=learning_rate)
                             for ac in self.actors]
        self.critic_optims = [optimizer(cr.parameters(), lr=learning_rate)
                              for cr in self.critics]
        self.sub_policy_num = sub_policy_num
        self.pool = Pool() if pool is None else pool
        self.replay_buffer = (Buffer(replay_size, replay_device)
                              if replay_buffer is None
                              else replay_buffer)

        # Distribute models to available devices.
        if available_devices is not None and available_devices:
            nets = self.actors + self.actor_targets + \
                   self.critics + self.critic_targets
            # Only actors and critics are related
            connections = {(i, i + self.sub_policy_num * 2): 1
                           for i in range(self.sub_policy_num)}
            # We do not need the assigner after construction.
            _assigner = ModelAssigner(nets, connections, available_devices)

            # For debugging:
            (act_assign, act_target_assign,
             critic_assign, critic_target_assign) = \
                np.array_split(_assigner.assignment, 4)
            default_logger.log("Actors assigned to:")
            default_logger.log(act_assign)
            default_logger.log("Actors (target) assigned to:")
            default_logger.log(act_target_assign)
            default_logger.log("Critics assigned to:")
            default_logger.log(critic_assign)
            default_logger.log("Critics (target) assigned to:")
            default_logger.log(critic_target_assign)

        # Create wrapper for target actors and target critics.
        # So their parameters can be saved.
        self.actor_target = nn.Module()
        self.critic_target = nn.Module()
        for actor_t, idx in zip(self.actor_targets,
                                range(self.sub_policy_num)):
            self.actor_target.add_module("actor_{}".format(idx), actor_t)

        for critic_t, idx in zip(self.critic_targets,
                                 range(self.sub_policy_num)):
            self.critic_target.add_module("critic_{}".format(idx), critic_t)

        # Make sure target and online networks have the same weight
        with t.no_grad():
            self.pool.starmap(hard_update,
                              zip(self.actors, self.actor_targets))
            self.pool.starmap(hard_update,
                              zip(self.critics, self.critic_targets))

        if lr_scheduler is not None:
            self.actor_lr_schs = [lr_scheduler(ac_opt,
                                               *lr_scheduler_args[0],
                                               *lr_scheduler_kwargs[0])
                                  for ac_opt in self.actor_optims]
            self.critic_lr_schs = [lr_scheduler(cr_opt,
                                                *lr_scheduler_args[1],
                                                *lr_scheduler_kwargs[1])
                                   for cr_opt in self.critic_optims]

        self.criterion = criterion

        self.reward_func = (MADDPG.bellman_function
                            if reward_func is None
                            else reward_func)

        self.action_alter_func = (MADDPG.action_alter_function
                                  if action_alter_func is None
                                  else action_alter_func)

        self.action_concat_func = (MADDPG.action_concat_function
                                   if action_concat_func is None
                                   else action_concat_func)

        self.state_split_func = (MADDPG.state_split_function
                                 if state_split_func is None
                                 else state_split_func)

        super(MADDPG, self).__init__()

    def act(self,
            state: Dict[str, Any],
            use_target: bool = False,
            index: int = -1,
            **__):
        """
        Use actor network to produce an action for the current state.

        Args:
            state: Current state.
            use_target: Whether use the target network.
            index: The sub-policy index to use.

        Returns:
            Action of shape ``[batch_size, action_dim]``.
        """
        if index not in range(self.sub_policy_num):
            index = np.random.randint(0, self.sub_policy_num)

        if use_target:
            return safe_call(self.actor_targets[index], state)
        else:
            return safe_call(self.actors[index], state)

    def act_with_noise(self,
                       state: Dict[str, Any],
                       noise_param: Tuple = (0.0, 1.0),
                       ratio: float = 1.0,
                       mode: str = "uniform",
                       use_target: bool = False,
                       index: int = -1,
                       **__):
        """
        Use actor network to produce a noisy action for the current state.

        See Also:
             :mod:`machin.frame.noise.action_space_noise`

        Args:
            state: Current state.
            noise_param: Noise params.
            ratio: Noise ratio.
            mode: Noise mode. Supported are:
                ``"uniform", "normal", "clipped_normal", "ou"``
            use_target: Whether use the target network.
            index: The sub-policy index to use.

        Returns:
            Noisy action of shape ``[batch_size, action_dim]``.
        """
        if mode == "uniform":
            return add_uniform_noise_to_action(
                self.act(state, use_target, index), noise_param, ratio
            )
        if mode == "normal":
            return add_normal_noise_to_action(
                self.act(state, use_target, index), noise_param, ratio
            )
        if mode == "clipped_normal":
            return add_clipped_normal_noise_to_action(
                self.act(state, use_target, index), noise_param, ratio
            )
        if mode == "ou":
            return add_ou_noise_to_action(
                self.act(state, use_target, index), noise_param, ratio
            )
        raise RuntimeError("Unknown noise type: " + str(mode))

    def act_discreet(self,
                     state: Dict[str, Any],
                     use_target: bool = False,
                     index: int = -1):
        """
        Use actor network to produce a discreet action for the current state.

        Notes:
            actor network must output a probability tensor, of shape
            (batch_size, action_dims), and has a sum of 1 for each row
            in dimension 1.

        Args:
            state: Current state.
            use_target: Whether to use the target network.
            index: The sub-policy index to use.

        Returns:
            Action of shape ``[batch_size, 1]``.
        """
        if index not in range(self.sub_policy_num):
            index = np.random.randint(0, self.sub_policy_num)

        if use_target:
            result = safe_call(self.actor_targets[index], state)
        else:
            result = safe_call(self.actors[index], state)

        assert_output_is_probs(result)
        batch_size = result.shape[0]
        result = t.argmax(result, dim=1).view(batch_size, 1)
        return result

    def act_discreet_with_noise(self,
                                state: Dict[str, Any],
                                use_target: bool = False,
                                index=-1):
        """
        Use actor network to produce a noisy discreet action for
        the current state.

        Notes:
            actor network must output a probability tensor, of shape
            (batch_size, action_dims), and has a sum of 1 for each row
            in dimension 1.

        Args:
            state: Current state.
            use_target: Whether to use the target network.
            index: The sub-policy index to use.

        Returns:
            Noisy action of shape ``[batch_size, 1]``.
        """
        if index not in range(self.sub_policy_num):
            index = np.random.randint(0, self.sub_policy_num)

        if use_target:
            result = safe_call(self.actor_targets[index], state)
        else:
            result = safe_call(self.actors[index], state)

        assert_output_is_probs(result)
        dist = Categorical(result)
        batch_size = result.shape[0]
        return dist.sample([batch_size, 1])

    def criticize(self, all_states, all_actions, use_target=False, index=-1):
        """
        Use critic network to evaluate current value.

        Args:
            all_states: Current states of all actors.
            all_actions: Current actions of all actors.
            use_target: Whether to use the target network.
            index: The sub-critic index to use.

        Returns:
            Value of shape ``[batch_size, 1]``.
        """
        if index not in range(self.sub_policy_num):
            index = np.random.randint(0, self.sub_policy_num)

        if use_target:
            return safe_call(self.critic_targets[index],
                             all_states, all_actions)
        else:
            return safe_call(self.critics[index],
                             all_states, all_actions)

    def store_transition(self, transition: Union[MultiAgentTransition, Dict]):
        """
        Add a transition sample to the replay buffer.
        """
        self.replay_buffer.append(transition, required_attrs=(
            "state", "all_states", "all_actions", "all_next_states",
            "reward", "terminal", "index"
        ))

    def store_episode(self, episode: List[Union[MultiAgentTransition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.
        """
        for trans in episode:
            self.replay_buffer.append(trans, required_attrs=(
                "state", "all_states", "all_actions", "all_next_states",
                "reward", "terminal", "index"
            ))

    def update(self,
               update_value=True,
               update_policy=True,
               update_target=True,
               concatenate_samples=True,
               average_target_parameter=False):
        """
        Update network weights by sampling from replay buffer.

        Args:
            update_value: Whether to update the Q network.
            update_policy: Whether to update the actor network.
            update_target: Whether to update targets.
            concatenate_samples: Whether to concatenate the samples.
            average_target_parameter: Whether to average sub target networks,
                including actors and critics.
        Returns:
            mean value of estimated policy value, value loss
        """
        batch_size, (state, all_states, all_actions, all_next_states,
                     reward, terminal, agent_indexes, *others) = \
            self.replay_buffer.sample_batch(self.batch_size,
                                            concatenate_samples,
                                            sample_attrs=[
                                                "state", "all_states",
                                                "all_actions",
                                                "all_next_states",
                                                "reward", "terminal", "index",
                                                "*"
                                            ])

        def update_inner(i):
            with t.no_grad():
                # Produce all_next_actions for all_next_states, using target i,
                # so the target critic can evaluate the value of the next step.
                all_next_actions_t = \
                    self.action_concat_func(
                        self.act(
                            self.state_split_func(
                                all_next_states, batch_size,
                                self.agent_num, others
                            ),
                            True, i
                        ),
                        batch_size, self.agent_num, others
                    )

            # Update critic network first
            # Generate target value using target critic.
            with t.no_grad():
                next_value = self.criticize(all_next_states,
                                            all_next_actions_t,
                                            True, i)
                next_value = next_value.view(batch_size, -1)
                y_i = self.reward_func(reward, self.discount, next_value,
                                       terminal, others)

            # action contain actions of all agents, same for state
            cur_value = self.criticize(all_states, all_actions, index=i)
            value_loss = self.criterion(cur_value, y_i.to(cur_value.device))

            if update_value:
                self.critics[i].zero_grad()
                value_loss.backward()
                self.critic_optims[i].step()

            # Update actor network
            cur_all_actions = copy.deepcopy(all_actions)
            cur_all_actions = self.action_alter_func(
                self.act(state, index=i), cur_all_actions, agent_indexes,
                batch_size, self.agent_num, others
            )
            act_value = self.criticize(state, cur_all_actions, index=i)

            # "-" is applied because we want to maximize J_b(u),
            # but optimizer workers by minimizing the target
            act_policy_loss = -act_value.mean()

            if update_policy:
                self.actors[i].zero_grad()
                act_policy_loss.backward()
                self.actor_optims[i].step()

            # Update target networks
            if update_target:
                soft_update(self.actor_targets[i], self.actors[i],
                            self.update_rate)
                soft_update(self.critic_targets[i], self.critics[i],
                            self.update_rate)

            return act_policy_loss.item(), value_loss.item()

        all_loss = self.pool.map(update_inner, range(self.sub_policy_num))
        mean_loss = t.tensor(all_loss).mean(dim=0)

        if average_target_parameter:
            self.average_target_parameters()

        # returns action value and policy loss
        return -mean_loss[0].item(), mean_loss[1].item()

    def update_lr_scheduler(self):
        """
        Update learning rate schedulers.
        """
        if hasattr(self, "actor_lr_schs"):
            for actor_lr_sch in self.actor_lr_schs:
                actor_lr_sch.step()
        if hasattr(self, "critic_lr_schs"):
            for critic_lr_sch in self.critic_lr_schs:
                critic_lr_sch.step()

    def load(self, model_dir, network_map=None, version=-1):
        # DOC INHERITED
        super(MADDPG, self).load(model_dir, network_map, version)
        with t.no_grad():
            self.pool.starmap(hard_update,
                              zip(self.actors, self.actor_targets))
            self.pool.starmap(hard_update,
                              zip(self.critics, self.critic_targets))

    def average_target_parameters(self):
        """
        Average parameters of sub-policies and sub-critics. Averaging
        is performed on target networks.
        """
        with t.no_grad():
            actor_params = [net.parameters() for net in self.actor_targets]
            critic_params = [net.parameters() for net in self.critic_targets]
            self.pool.starmap(
                _average_parameters,
                itertools.chain(zip(*actor_params), zip(*critic_params))
            )

    @staticmethod
    def action_alter_function(raw_output_action, all_actions, indexes,
                              batch_size, agent_num, *_):
        """
        This function is used to alternate an action inside all actions,
        using output from the online actor network.

        Args:
            raw_output_action: Raw output of actor.
            all_actions: All actions of all agents.
            indexes: Agent index among all agents.
            batch_size: Sampled batch size.
            agent_num: Number of agents.

        Returns:
            Alternated all actions.
        """
        all_actions["action"][indexes] = \
            raw_output_action.view(batch_size, agent_num, -1)
        return all_actions

    @staticmethod
    def state_split_function(all_states: Dict[str, t.Tensor],
                             batch_size: int,
                             agent_num: int,
                             *_):
        """
        This function is used to split states from multiple agents into
        batched single states, usable by actor network.

        Args:
            all_states: All states of all agents.
            batch_size: Sampled batch size.
            agent_num: Number of agents.
        Returns:
            Splitted states.
        """
        all_states["state"] = all_states["state"]\
            .view(batch_size * agent_num, -1)
        return all_states

    @staticmethod
    def action_concat_function(raw_output_action, batch_size, agent_num, *_):
        """
        This function is used to transform the actions produced by actor
        from the splitted states, to the final output of all actions of
        all agents.

        Args:
            raw_output_action: Raw output of actor.
            batch_size: Sampled batch size.
            agent_num: Number of agents.

        Returns:
            Concatenated actions.
        """
        return {"action": raw_output_action.view(batch_size, agent_num)}

    @staticmethod
    def bellman_function(reward, discount, next_value, terminal, *_):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * (1 - terminal) * next_value
