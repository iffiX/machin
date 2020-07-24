import copy
import itertools
from random import choice, randint
from machin.parallel.pool import Pool
# pylint: disable=wildcard-import, unused-wildcard-import
from .ddpg import *


class SHMBuffer(Buffer):
    @staticmethod
    def make_tensor_from_batch(batch, device, concatenate):
        # this function is used in post processing, and we will
        # move all cpu tensors to shared memory.
        if concatenate and len(batch) != 0:
            item = batch[0]
            batch_size = len(batch)
            if t.is_tensor(item):
                batch = [it.to(device) for it in batch]
                result = t.cat(batch, dim=0).to(device)
                result.share_memory_()
                return result
            else:
                try:
                    result = t.tensor(batch, device=device)\
                        .view(batch_size, -1)
                    result.share_memory_()
                    return result
                except Exception:
                    raise ValueError("Batch not concatenable: {}"
                                     .format(batch))
        else:
            for it in batch:
                if t.is_tensor(it):
                    it.share_memory_()
            return batch


class MADDPG(TorchFramework):
    """
    MADDPG is a centralized multi-agent training framework, it alleviates the
    unstable reward problem caused by the disturbance of other agents by
    gathering all agents observations and train a global critic. This global
    critic observes all actions and all states from all agents.
    """
    # Since the number of sub-policies is automatically determined,
    # they are not considered here.
    _is_top = ["all_actor_target", "all_critic_target"]
    _is_restorable = ["all_actor_target", "all_critic_target"]

    def __init__(self,
                 actors: List[Union[NeuralNetworkModule, nn.Module]],
                 actor_targets: List[Union[NeuralNetworkModule, nn.Module]],
                 critics: List[Union[NeuralNetworkModule, nn.Module]],
                 critic_targets: List[Union[NeuralNetworkModule, nn.Module]],
                 optimizer: Callable,
                 criterion: Callable,
                 *_,
                 sub_policy_num: int = 0,
                 lr_scheduler: Callable = None,
                 lr_scheduler_args: Tuple[Tuple, Tuple] = None,
                 lr_scheduler_kwargs: Tuple[Dict, Dict] = None,
                 batch_size: int = 100,
                 update_rate: float = 0.001,
                 actor_learning_rate: float = 0.0005,
                 critic_learning_rate: float = 0.001,
                 discount: float = 0.99,
                 gradient_max: float = np.inf,
                 replay_size: int = 500000,
                 replay_device: Union[str, t.device] = "cpu",
                 replay_buffer: Buffer = None,
                 reward_func: Callable = None,
                 action_trans_func: Callable = None,
                 action_concat_func: Callable = None,
                 state_concat_func: Callable = None,
                 visualize: bool = False,
                 visualize_dir: str = "",
                 pool_size: int = None):
        """
        See Also:
            :class:`.DDPG`

        Note:
            In order to parallelize agent inference, a process pool is used
            internally. However, in order to minimize memory copy / CUDA memory
            copy, the location of all of your models must be either "cpu", or
            "cuda" (Using multiple CUDA devices is supported).

        Note:
            MADDPG framework assumes that all of your actors are homogeneous,
            and critics are homogeneous as well.

        Note:
            This implementation contains:
                - Ensemble Training

            This implementation doe not contain:
                - Inferring other agents' policies
                - Mixed continuous/discrete action spaces

        Args:
            actors: Actor network modules.
            actor_targets: Target actor network modules.
            critics: Critic network modules.
            critic_targets: Target critic network modules.
            optimizer: Optimizer used to optimize ``actors`` and ``critics``.
            criterion: Criterion used to evaluate the value loss.
            sub_policy_num: Times to replicate each actor. Equals to
                `ensemble_policy_num - 1`
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            batch_size: Batch size used during training.
            update_rate: :math:`\\tau` used to update target networks.
                Target parameters are updated as:
                :math:`\\theta_t = \\theta * \\tau + \\theta_t * (1 - \\tau)`
            actor_learning_rate: Learning rate of the actor optimizer,
                not compatible with ``lr_scheduler``.
            critic_learning_rate: Learning rate of the critic optimizer,
                not compatible with ``lr_scheduler``.
            discount: :math:`\\gamma` used in the bellman function.
            replay_size: Replay buffer size for each actor. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer. Will be replicated for actor.
            reward_func: Reward function used in training.
            action_trans_func: Action transform function, used to transform
                the raw output of your actor.
            action_concat_func: Action concatenation function, used to
                concatenate a list of action dicts into one dict.
            state_concat_func: State concatenation function, used to
                concatenate a list of state dicts into one dict.
            visualize: Whether visualize the network flow in the first pass.
            visualize_dir: Visualized graph save directory.
            pool_size: Size of the internal execution pool.
        """
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.discount = discount
        self.visualize = visualize
        self.visualize_dir = visualize_dir
        self.grad_max = gradient_max

        # create ensembles of policies
        self.actors = [[actor] +
                       [copy.deepcopy(actor) for _ in range(sub_policy_num)]
                       for actor in actors]
        self.actor_targets = [[actor_target] +
                              [copy.deepcopy(actor_target)
                               for _ in range(sub_policy_num)]
                              for actor_target in actor_targets]
        self.critics = critics
        self.critic_targets = critic_targets
        self.actor_optims = [[optimizer(acc.parameters(),
                                        lr=actor_learning_rate)
                             for acc in ac]
                             for ac in self.actors]
        self.critic_optims = [optimizer(cr.parameters(),
                                        lr=critic_learning_rate)
                              for cr in self.critics]
        self.ensemble_size = sub_policy_num + 1
        self.replay_buffers = [SHMBuffer(replay_size, replay_device)
                               if replay_buffer is None
                               else copy.deepcopy(replay_buffer)
                               for _ in range(len(actors))]

        # check devices of all parameters,
        # determine the pool process starting method.
        device = self._check_parameters_device(
            itertools.chain(*self.actors, self.critics)
        )
        self.device = device
        self.pool = Pool(processes=pool_size,
                         is_recursive=False,
                         is_copy_tensor=False,
                         share_method=device)

        # Create wrapper for target actors and target critics.
        # So their parameters can be saved.
        self.all_actor_target = nn.Module()
        self.all_critic_target = nn.Module()

        for ac, idx in zip(self.actor_targets, range(len(actors))):
            for acc, idxx in zip(ac, range(self.ensemble_size)):
                acc.share_memory()
                self.all_actor_target.add_module(
                    "actor_{}_{}".format(idx, idxx), acc
                )

        for cr, idx in zip(self.critic_targets, range(len(critics))):
            cr.share_memory()
            self.all_critic_target.add_module(
                "critic_{}".format(idx), cr
            )

        # Make sure target and online networks have the same weight
        with t.no_grad():
            self.pool.starmap(hard_update,
                              zip(itertools.chain(*self.actors),
                                  itertools.chain(*self.actor_targets)))
            self.pool.starmap(hard_update,
                              zip(self.critics, self.critic_targets))

        if lr_scheduler is not None:
            if lr_scheduler_args is None:
                lr_scheduler_args = ((), ())
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = ({}, {})
            self.actor_lr_schs = [lr_scheduler(ac_opt,
                                               *lr_scheduler_args[0],
                                               *lr_scheduler_kwargs[0])
                                  for acc_opt in self.actor_optims
                                  for ac_opt in acc_opt]
            self.critic_lr_schs = [lr_scheduler(cr_opt,
                                                *lr_scheduler_args[1],
                                                *lr_scheduler_kwargs[1])
                                   for cr_opt in self.critic_optims]

        self.criterion = criterion

        self.reward_func = (MADDPG.bellman_function
                            if reward_func is None
                            else reward_func)
        self.action_trans_func = (MADDPG.action_transform_function
                                  if action_trans_func is None
                                  else action_trans_func)
        self.action_concat_func = (MADDPG.action_concat_function
                                   if action_concat_func is None
                                   else action_concat_func)
        self.state_concat_func = (MADDPG.state_concat_function
                                  if state_concat_func is None
                                  else state_concat_func)

        super(MADDPG, self).__init__()

    def act(self,
            states: List[Dict[str, Any]],
            use_target: bool = False,
            **__):
        """
        Use all actor networks to produce actions for the current state.
        A random sub-policy from the policy ensemble of each actor will
        be chosen.

        Args:
            states: A list of current states of each actor.
            use_target: Whether use the target network.

        Returns:
            A list of actions of shape ``[batch_size, action_dim]``.
        """
        if use_target:
            actors = [choice(sub_actors) for sub_actors in self.actor_targets]
            return self.pool.starmap(self._no_grad_safe_call,
                                     zip(actors, states))
        else:
            actors = [choice(sub_actors) for sub_actors in self.actors]
            return self.pool.starmap(self._no_grad_safe_call,
                                     zip(actors, states))

    def act_with_noise(self,
                       states: List[Dict[str, Any]],
                       noise_param: Any = (0.0, 1.0),
                       ratio: float = 1.0,
                       mode: str = "uniform",
                       use_target: bool = False,
                       **__):
        """
        Use all actor networks to produce noisy actions for the current state.
        A random sub-policy from the policy ensemble of each actor will
        be chosen.

        See Also:
             :mod:`machin.frame.noise.action_space_noise`

        Args:
            states: A list of current states of each actor.
            noise_param: Noise params.
            ratio: Noise ratio.
            mode: Noise mode. Supported are:
                ``"uniform", "normal", "clipped_normal", "ou"``
            use_target: Whether use the target network.

        Returns:
            A list of noisy actions of shape ``[batch_size, action_dim]``.
        """
        actions = self.act(states, use_target)

        if mode == "uniform":
            return [add_uniform_noise_to_action(act, noise_param, ratio)
                    for act in actions]
        if mode == "normal":
            return [add_normal_noise_to_action(act, noise_param, ratio)
                    for act in actions]
        if mode == "clipped_normal":
            return [add_clipped_normal_noise_to_action(act, noise_param, ratio)
                    for act in actions]
        if mode == "ou":
            return [add_ou_noise_to_action(act, noise_param, ratio)
                    for act in actions]
        raise ValueError("Unknown noise type: " + str(mode))

    def act_discreet(self,
                     states: List[Dict[str, Any]],
                     use_target: bool = False):
        """
        Use all actor networks to produce discreet actions for the current
        state.
        A random sub-policy from the policy ensemble of each actor will
        be chosen.

        Notes:
            actor network must output a probability tensor, of shape
            (batch_size, action_dims), and has a sum of 1 for each row
            in dimension 1.

        Args:
            states: A list of current states of each actor.
            use_target: Whether use the target network.

        Returns:
            A list of discreet actions of shape ``[batch_size, 1]``.
            A list of action probability tensors of shape
            ``[batch_size, action_num]``.
        """
        actions = self.act(states, use_target)
        result = []
        for action in actions:
            assert_output_is_probs(action)
            batch_size = action.shape[0]
            result.append(t.argmax(action, dim=1).view(batch_size, 1))
        return result, actions

    def act_discreet_with_noise(self,
                                states: List[Dict[str, Any]],
                                use_target: bool = False):
        """
        Use all actor networks to produce discreet actions for the current
        state.
        A random sub-policy from the policy ensemble of each actor will
        be chosen.

        Notes:
            actor network must output a probability tensor, of shape
            (batch_size, action_dims), and has a sum of 1 for each row
            in dimension 1.

        Args:
            states: A list of current states of each actor.
            use_target: Whether use the target network.

        Returns:
            A list of noisy discreet actions of shape ``[batch_size, 1]``.
            A list of action probability tensors of shape
            ``[batch_size, action_num]``.
        """
        actions = self.act(states, use_target)
        result = []
        for action in actions:
            assert_output_is_probs(action)
            batch_size = action.shape[0]
            dist = Categorical(action)
            result.append(dist.sample([batch_size, 1]))
        return result, actions

    def criticize(self,
                  states: List[Dict[str, Any]],
                  actions: List[Dict[str, Any]],
                  index: int,
                  use_target=False):
        """
        Use critic network to evaluate current value.

        Args:
            states: Current states of all actors.
            actions: Current actions of all actors.
            use_target: Whether to use the target network.
            index: Index of the used critic.

        Returns:
            Value of shape ``[batch_size, 1]``.
        """
        if use_target:
            return safe_call(self.critic_targets[index],
                             self.state_concat_func(states),
                             self.action_concat_func(actions))
        else:
            return safe_call(self.critics[index],
                             self.state_concat_func(states),
                             self.action_concat_func(actions))

    def store_transitions(self, transitions: List[Union[Transition, Dict]]):
        """
        Add a list of transition samples, from all actors at the same time
        step, to the replay buffers.

        Args:
            transitions: List of transition objects.
        """
        assert len(transitions) == len(self.replay_buffers)
        for buff, trans in zip(self.replay_buffers, transitions):
            buff.append(trans, required_attrs=(
                "state", "action", "next_state", "reward", "terminal"
            ))

    def store_episodes(self, episodes: List[List[Union[Transition, Dict]]]):
        """
        Add a List of full episodes, from all actors, to the replay buffers.
        Each episode is a list of transition samples.
        """
        assert len(episodes) == len(self.replay_buffers)
        all_length = [len(ep) for ep in episodes]
        assert len(set(all_length)) == 1, \
            "All episodes must have the same length!"
        for buff, ep in zip(self.replay_buffers, episodes):
            for trans in ep:
                buff.append(trans, required_attrs=(
                    "state", "action", "next_state", "reward", "terminal"
                ))

    def update(self,
               update_value=True,
               update_policy=True,
               update_target=True,
               concatenate_samples=True):
        """
        Update network weights by sampling from replay buffer.

        Args:
            update_value: Whether to update the Q network.
            update_policy: Whether to update the actor network.
            update_target: Whether to update targets.
            concatenate_samples: Whether to concatenate the samples.
        Returns:
            mean value of estimated policy value, value loss
        """
        # All buffers should have the same length now.

        # Create a sample method per update
        # this sample method will sample the same indexes
        # (different for each update() call) on all buffers.
        buffer_length = self.replay_buffers[0].size()
        if buffer_length == 0:
            return
        batch_size = min(buffer_length, self.batch_size)
        sample_indexes = [[randint(0, buffer_length - 1)
                           for _ in range(batch_size)]
                          for __ in range(self.ensemble_size)]

        sample_methods = [self._create_sample_method(indexes)
                          for indexes in sample_indexes]

        # Now sample from buffer for each sub-policy in the ensemble.
        # To reduce memory usage, for each sub-policy "i" of each actor,
        # the same sample "i" will be used for training.

        # Tensors in the sampled batch will be moved to shared memory.

        # size: [ensemble size, num of actors]
        batches = []
        for e_idx in range(self.ensemble_size):
            ensemble_batch = []
            for a_idx in range(len(self.actors)):
                batch_size_, batch = \
                    self.replay_buffers[a_idx].sample_batch(
                        self.batch_size, concatenate_samples,
                        sample_method=sample_methods[e_idx],
                        sample_attrs=[
                            "state", "action", "reward", "next_state",
                            "terminal", "*"]
                    )
                ensemble_batch.append(batch)
                assert batch_size_ == batch_size
            batches.append(ensemble_batch)

        args = []
        for e_idx in range(self.ensemble_size):
            for a_idx in range(len(self.actors)):
                args.append((
                    batch_size, batches, a_idx, e_idx,
                    self.actors, self.actor_targets,
                    self.critics, self.critic_targets,
                    self.actor_optims, self.critic_optims,
                    update_value, update_policy, update_target,
                    self.action_trans_func, self.action_concat_func,
                    self.state_concat_func, self.reward_func,
                    self.criterion, self.discount, self.update_rate,
                    self.grad_max
                ))

        if not self.visualize:
            all_loss = self.pool.starmap(self._update_sub_policy, args)
        else:
            all_loss = [[0.0, 0.0]]
            # Since it is hard to keep track of visualized parts
            # in a process pool (hard to share states between processes )
            # Only visualize the first actor-critic pair
            self._update_sub_policy(
                *args[0],
                visualize=True,
                visualize_func=self.visualize_model,
                visualize_dir=self.visualize_dir
            )

        mean_loss = t.tensor(all_loss).mean(dim=0)

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
                              zip(itertools.chain(*self.actors),
                                  itertools.chain(*self.actor_targets)))
            self.pool.starmap(hard_update,
                              zip(self.critics, self.critic_targets))

    @staticmethod
    def _no_grad_safe_call(*args, **kwargs):
        with t.no_grad():
            return safe_call(*args, **kwargs).detach()

    @staticmethod
    def _update_sub_policy(batch_size, batches, actor_index, policy_index,
                           actors, actor_targets, critics, critic_targets,
                           actor_optims, critic_optims,
                           update_value, update_policy, update_target,
                           atf, acf, scf, rf,
                           criterion, discount, update_rate, grad_max,
                           visualize=False, visualize_func=None,
                           visualize_dir=None):
        # atf: action transform function, used to transform the
        #      raw output of a single actor to a arg dict like:
        #      {"action": tensor}, where "action" is the keyword argument
        #      name of the critic.
        #
        # acf: action concatenation function, used to concatenate
        #      a list of action dicts into a single arg dict readable
        #      by critic.
        # scf: state concatenation function, used to concatenate
        #      a list of state dicts into a single arg dict readable
        #      by critic.
        # rf: reward function

        # The innermost element of ``batches``:
        # (state, action, reward, next_state, terminal, *)
        # ``batches`` size: [ensemble_size, actor_num]
        # select the batch for this sub-policy in the ensemble
        ensemble_batch = batches[policy_index]

        with t.no_grad():
            # Produce all_next_actions for all_next_states, using target i,
            # so the target critic can evaluate the value of the next step.
            all_next_actions_t = [
                atf(safe_call(choice(actor_targets[a_idx]),
                              ensemble_batch[a_idx][3]),
                    ensemble_batch[a_idx][5])
                if a_idx != actor_index else
                atf(safe_call(actor_targets[a_idx][policy_index],
                              ensemble_batch[a_idx][3]),
                    ensemble_batch[a_idx][5])
                for a_idx in range(len(actors))
            ]
            all_next_actions_t = acf(all_next_actions_t)

            cur_all_actions = [
                batch[1] for batch in ensemble_batch
            ]
            cur_all_actions = acf(cur_all_actions)

            all_next_states = [
                batch[3] for batch in ensemble_batch
            ]
            all_next_states = scf(all_next_states)

            all_states = [
                batch[0] for batch in ensemble_batch
            ]
            all_states = scf(all_states)

        # Update critic network first
        # Generate target value using target critic.
        with t.no_grad():
            reward = ensemble_batch[actor_index][2]
            terminal = ensemble_batch[actor_index][4]
            next_value = safe_call(critic_targets[actor_index],
                                   all_next_states,
                                   all_next_actions_t)
            next_value = next_value.view(batch_size, -1)
            y_i = rf(reward, discount, next_value, terminal,
                     ensemble_batch[actor_index][5])

        cur_value = safe_call(critics[actor_index],
                              all_states,
                              cur_all_actions)
        value_loss = criterion(cur_value, y_i.to(cur_value.device))

        if visualize:
            # only invoked if not running by pool
            visualize_func(value_loss, "critic", visualize_dir)

        if update_value:
            critics[actor_index].zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                critics[actor_index].parameters(), grad_max
            )
            critic_optims[actor_index].step()

        # Update actor network
        all_actions = [
            batch[1] for batch in ensemble_batch
        ]
        all_actions[actor_index] = atf(safe_call(
            actors[actor_index][policy_index],
            ensemble_batch[actor_index][3]
        ), ensemble_batch[actor_index][5])
        all_actions = acf(all_actions)
        act_value = safe_call(critics[actor_index],
                              all_states,
                              all_actions)

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()

        if visualize:
            # only invoked if not running by pool
            visualize_func(act_policy_loss, "actor", visualize_dir)

        if update_policy:
            actors[actor_index][policy_index].zero_grad()
            act_policy_loss.backward()
            nn.utils.clip_grad_norm_(
                actors[actor_index][policy_index].parameters(), grad_max
            )
            actor_optims[actor_index][policy_index].step()

        # Update target networks
        if update_target:
            soft_update(actor_targets[actor_index][policy_index],
                        actors[actor_index][policy_index],
                        update_rate)
            soft_update(critic_targets[actor_index],
                        critics[actor_index],
                        update_rate)

        return -act_policy_loss.item(), value_loss.item()

    @staticmethod
    def _check_parameters_device(models):
        devices = set()
        for model in models:
            for k, v in model.named_parameters():
                devices.add(v.device.type)
                if len(devices) > 1:
                    raise RuntimeError("All of your models should either"
                                       "locate on GPUs or on your CPU!")
        return list(devices)[0]

    @staticmethod
    def _create_sample_method(indexes):
        def sample_method(buffer, _len):
            nonlocal indexes
            batch = [buffer[i] for i in indexes
                     if i < len(buffer)]
            return len(batch), batch
        return sample_method

    @staticmethod
    def action_transform_function(raw_output_action: Any, *_):
        return {"action": raw_output_action}

    @staticmethod
    def action_concat_function(actions: List[Dict], *_):
        # Assume an atom action is [batch_size, action_dim]
        # concatenate actions in the second dimension.
        # becomes [batch_size, actor_num * action_dim]
        keys = actions[0].keys()
        all_actions = {}
        for k in keys:
            all_actions[k] = t.cat([act[k] for act in actions], dim=1)
        return all_actions

    @staticmethod
    def state_concat_function(states: List[Dict], *_):
        # Assume an atom state is [batch_size, state_dim]
        # concatenate states in the second dimension.
        # becomes [batch_size, actor_num * state_dim]
        keys = states[0].keys()
        all_states = {}
        for k in keys:
            all_states[k] = t.cat([st[k] for st in states], dim=1)
        return all_states

    @staticmethod
    def bellman_function(reward, discount, next_value, terminal, *_):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * ~terminal * next_value
