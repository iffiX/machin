import copy
import inspect
import itertools
from random import choice, randint
from .utils import determine_device
from machin.utils.visualize import make_dot
from machin.utils.logging import default_logger
from machin.model.nets.base import static_module_wrapper
from machin.parallel.pool import P2PPool, ThreadPool

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
                    result = t.tensor(batch, device=device).view(batch_size, -1)
                    result.share_memory_()
                    return result
                except Exception:
                    raise ValueError(f"Batch not concatenable: {batch}")
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

    def __init__(
        self,
        actors: List[Union[NeuralNetworkModule, nn.Module]],
        actor_targets: List[Union[NeuralNetworkModule, nn.Module]],
        critics: List[Union[NeuralNetworkModule, nn.Module]],
        critic_targets: List[Union[NeuralNetworkModule, nn.Module]],
        optimizer: Callable,
        criterion: Callable,
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[List[Tuple], List[Tuple]] = None,
        lr_scheduler_kwargs: Tuple[List[Dict], List[Dict]] = None,
        critic_visible_actors: List[List[int]] = None,
        sub_policy_num: int = 0,
        batch_size: int = 100,
        update_rate: float = 0.001,
        update_steps: Union[int, None] = None,
        actor_learning_rate: float = 0.0005,
        critic_learning_rate: float = 0.001,
        discount: float = 0.99,
        gradient_max: float = np.inf,
        replay_size: int = 500000,
        replay_device: Union[str, t.device] = "cpu",
        replay_buffer: Buffer = None,
        visualize: bool = False,
        visualize_dir: str = "",
        use_jit: bool = True,
        pool_type: str = "thread",
        pool_size: int = None,
        **__,
    ):
        """
        See Also:
            :class:`.DDPG`

        Note:
            In order to parallelize agent inference, a process pool is used
            internally. However, in order to minimize memory copy / CUDA memory
            copy, the location of all of your models must be either "cpu", or
            "cuda" (Using multiple CUDA devices is supported).

        Note:
            MADDPG framework **does not require** all of your actors are
            homogeneous. Each pair of your actors and critcs could be
            heterogeneous.

        Note:
            Suppose you have three pair of actors and critics, with index 0, 1,
            2. If critic 0 can observe the action of actor 0 and 1, critic 1 can
            observe the action of actor 1 and 2, critic 2 can observe the action
            of actor 2 and 0, the ``critic_visible_actors`` should be::

                [[0, 1], [1, 2], [2, 0]]

        Note:
            Learning rate scheduler args and kwargs for each actor and critic,
            the first list is for actors, and the second list is for critics.

        Note:
            This implementation contains:
                - Ensemble Training

            This implementation does not contain:
                - Inferring other agents' policies
                - Mixed continuous/discrete action spaces

        Args:
            actors: Actor network modules.
            actor_targets: Target actor network modules.
            critics: Critic network modules.
            critic_targets: Target critic network modules.
            optimizer: Optimizer used to optimize ``actors`` and ``critics``.
                By default all critics can see outputs of all actors.
            criterion: Criterion used to evaluate the value loss.
            critic_visible_actors: Indexes of visible actors for each critic.
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
            update_steps: Training step number used to update target networks.
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
            visualize: Whether visualize the network flow in the first pass.
            visualize_dir: Visualized graph save directory.
            use_jit: Whether use torch jit to perform the forward pass
                in parallel instead of using the internal pool. Provides
                significant speed and efficiency advantage, but requires
                actors and critics convertible to TorchScript.
            pool_type: Type of the internal execution pool, either "process"
                or "thread".
            pool_size: Size of the internal execution pool.
        """
        assert pool_type in ("process", "thread")
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.update_steps = update_steps
        self.discount = discount
        self.has_visualized = False
        self.visualize = visualize
        self.visualize_dir = visualize_dir
        self.grad_max = gradient_max
        self.critic_visible_actors = critic_visible_actors or [
            list(range(len(actors)))
        ] * len(actors)
        self._update_counter = 0

        if update_rate is not None and update_steps is not None:
            raise ValueError(
                "You can only specify one target network update"
                " scheme, either by update_rate or update_steps,"
                " but not both."
            )

        # create ensembles of policies
        self.actors = [
            [actor] + [copy.deepcopy(actor) for _ in range(sub_policy_num)]
            for actor in actors
        ]
        self.actor_targets = [
            [actor_target]
            + [copy.deepcopy(actor_target) for _ in range(sub_policy_num)]
            for actor_target in actor_targets
        ]
        self.critics = critics
        self.critic_targets = critic_targets
        self.actor_optims = [
            [optimizer(acc.parameters(), lr=actor_learning_rate) for acc in ac]
            for ac in self.actors
        ]
        self.critic_optims = [
            optimizer(cr.parameters(), lr=critic_learning_rate) for cr in self.critics
        ]
        self.ensemble_size = sub_policy_num + 1
        self.replay_buffers = [
            SHMBuffer(replay_size, replay_device)
            if replay_buffer is None
            else copy.deepcopy(replay_buffer)
            for _ in range(len(actors))
        ]

        # create the pool used to update()
        # check devices of all parameters,
        # determine the pool process starting method.
        device = self._check_parameters_device(
            itertools.chain(*self.actors, self.critics)
        )
        self.device = device

        self.pool_type = pool_type
        if pool_type == "process":
            self.pool = P2PPool(
                processes=pool_size,
                is_recursive=False,
                is_copy_tensor=False,
                share_method=device,
            )
        elif pool_type == "thread":
            self.pool = ThreadPool(processes=pool_size)

        # Create wrapper for target actors and target critics.
        # So their parameters can be saved.
        self.all_actor_target = nn.Module()
        self.all_critic_target = nn.Module()

        for ac, idx in zip(self.actor_targets, range(len(actors))):
            for acc, idxx in zip(ac, range(self.ensemble_size)):
                acc.share_memory()
                self.all_actor_target.add_module(f"actor_{idx}_{idxx}", acc)

        for cr, idx in zip(self.critic_targets, range(len(critics))):
            cr.share_memory()
            self.all_critic_target.add_module(f"critic_{idx}", cr)

        # Make sure target and online networks have the same weight
        with t.no_grad():
            self.pool.starmap(
                hard_update,
                zip(
                    itertools.chain(*self.actors), itertools.chain(*self.actor_targets)
                ),
            )
            self.pool.starmap(hard_update, zip(self.critics, self.critic_targets))

        if lr_scheduler is not None:
            if lr_scheduler_args is None:
                lr_scheduler_args = ([()] * len(actors), [()] * len(critics))
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = ([{}] * len(actors), [{}] * len(critics))
            self.actor_lr_schs = [
                lr_scheduler(acc_opt, *lr_sch_args, *lr_sch_kwargs)
                for ac_opt, lr_sch_args, lr_sch_kwargs in zip(
                    self.actor_optims, lr_scheduler_args[0], lr_scheduler_kwargs[0]
                )
                for acc_opt in ac_opt
            ]
            self.critic_lr_schs = [
                lr_scheduler(cr_opt, *lr_sch_args, *lr_sch_kwargs)
                for cr_opt, lr_sch_args, lr_sch_kwargs in zip(
                    self.critic_optims, lr_scheduler_args[1], lr_scheduler_kwargs[1]
                )
            ]

        self.criterion = criterion

        # make preparations if use jit
        # jit modules will share the same parameter memory with original
        # modules, therefore it is safe to use them together.
        self.use_jit = use_jit
        self.jit_actors = []
        self.jit_actor_targets = []
        if use_jit:
            # only compile actors, since critics will not be
            # launched in parallel
            for ac in self.actors:
                jit_actors = []
                jit_actor_targets = []
                for acc in ac:
                    # exclude "self" by truncating element 0
                    actor_arg_spec = inspect.getfullargspec(acc.forward)
                    jit_actor = t.jit.script(acc)
                    jit_actor.arg_spec = actor_arg_spec
                    jit_actor.model_type = type(acc)
                    jit_actors.append(jit_actor)

                    jit_actor_target = t.jit.script(acc)
                    jit_actor_target.arg_spec = actor_arg_spec
                    jit_actor_target.model_type = type(acc)
                    jit_actor_targets.append(jit_actor_target)
                self.jit_actors.append(jit_actors)
                self.jit_actor_targets.append(jit_actor_targets)

        super().__init__()

    @property
    def optimizers(self):
        return sum(self.actor_optims, self.critic_optims)

    @optimizers.setter
    def optimizers(self, optimizers):
        counter = 0
        for ac in self.actor_optims:
            for id, _acc in enumerate(ac):
                ac[id] = optimizers[counter]
                counter += 1
        for id in range(len(self.critic_optims)):
            self.critic_optims[id] = optimizers[counter]
            counter += 1

    @property
    def lr_schedulers(self):
        if hasattr(self, "actor_lr_schs") and hasattr(self, "critic_lr_schs"):
            return self.actor_lr_schs + self.critic_lr_schs
        return []

    def act(self, states: List[Dict[str, Any]], use_target: bool = False, **__):
        """
        Use all actor networks to produce actions for the current state.
        A random sub-policy from the policy ensemble of each actor will
        be chosen.

        Args:
            states: A list of current states of each actor.
            use_target: Whether use the target network.

        Returns:
            A list of anything returned by your actor. If your actor
            returns multiple values, they will be wrapped in a tuple.
        """
        return [safe_return(act) for act in self._act_api_general(states, use_target)]

    def act_with_noise(
        self,
        states: List[Dict[str, Any]],
        noise_param: Any = (0.0, 1.0),
        ratio: float = 1.0,
        mode: str = "uniform",
        use_target: bool = False,
        **__,
    ):
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
        actions = self._act_api_general(states, use_target)
        result = []
        for action, *others in actions:
            if mode == "uniform":
                action = add_uniform_noise_to_action(action, noise_param, ratio)
            elif mode == "normal":
                action = add_normal_noise_to_action(action, noise_param, ratio)
            elif mode == "clipped_normal":
                action = add_clipped_normal_noise_to_action(action, noise_param, ratio)
            elif mode == "ou":
                action = add_ou_noise_to_action(action, noise_param, ratio)
            else:
                raise ValueError("Unknown noise type: " + str(mode))
            if len(others) == 0:
                result.append(action)
            else:
                result.append((action, *others))
        return result

    def act_discrete(self, states: List[Dict[str, Any]], use_target: bool = False):
        """
        Use all actor networks to produce discrete actions for the current
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
            A list of tuples containing:
            1. Integer discrete actions of shape ``[batch_size, 1]``.
            2. Action probability tensors of shape ``[batch_size, action_num]``.
            3. Any other things returned by your actor.
        """
        actions = self._act_api_general(states, use_target)
        result = []
        for action, *others in actions:
            assert_output_is_probs(action)
            batch_size = action.shape[0]
            action_disc = t.argmax(action, dim=1).view(batch_size, 1)
            result.append((action_disc, action, *others))
        return result

    def act_discrete_with_noise(
        self, states: List[Dict[str, Any]], use_target: bool = False
    ):
        """
        Use all actor networks to produce discrete actions for the current
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
            A list of tuples containing:
            1. Integer noisy discrete actions.
            2. Action probability tensors of shape ``[batch_size, action_num]``.
            3. Any other things returned by your actor.
        """
        actions = self._act_api_general(states, use_target)
        result = []
        for action, *others in actions:
            assert_output_is_probs(action)
            batch_size = action.shape[0]
            dist = Categorical(action)
            action_disc = dist.sample([batch_size, 1]).view(batch_size, 1)
            result.append((action_disc, action, *others))
        return result

    def _act_api_general(self, states, use_target):
        if self.use_jit:
            if use_target:
                actors = [choice(sub_actors) for sub_actors in self.jit_actor_targets]
            else:
                actors = [choice(sub_actors) for sub_actors in self.jit_actors]
            future = [self._jit_safe_call(ac, st) for ac, st in zip(actors, states)]
            result = [t.jit._wait(fut) for fut in future]
            result = [res if isinstance(res, tuple) else (res,) for res in result]
        else:
            if use_target:
                actors = [choice(sub_actors) for sub_actors in self.actor_targets]
            else:
                actors = [choice(sub_actors) for sub_actors in self.actors]
            result = self.pool.starmap(self._no_grad_safe_call, zip(actors, states))
            result = [res for res in result]
        return result

    def _criticize(
        self,
        states: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        index: int,
        use_target=False,
    ):
        """
        Use critic network to evaluate current value.

        Args:
            states: Current states of all actors.
            actions: Current actions of all actors.
            use_target: Whether to use the target network.
            index: Index of the used critic.

        Returns:
            Q Value of shape ``[batch_size, 1]``.
        """
        if use_target:
            return safe_call(
                self.critic_targets[index],
                self.state_concat_function(states),
                self.action_concat_function(actions),
            )
        else:
            return safe_call(
                self.critics[index],
                self.state_concat_function(states),
                self.action_concat_function(actions),
            )

    def store_transitions(self, transitions: List[Union[Transition, Dict]]):
        """
        Add a list of transition samples, from all actors at the same time
        step, to the replay buffers.

        Args:
            transitions: List of transition objects.
        """
        assert len(transitions) == len(self.replay_buffers)
        for buff, trans in zip(self.replay_buffers, transitions):
            buff.append(
                trans,
                required_attrs=("state", "action", "next_state", "reward", "terminal"),
            )

    def store_episodes(self, episodes: List[List[Union[Transition, Dict]]]):
        """
        Add a List of full episodes, from all actors, to the replay buffers.
        Each episode is a list of transition samples.
        """
        assert len(episodes) == len(self.replay_buffers)
        all_length = [len(ep) for ep in episodes]
        assert len(set(all_length)) == 1, "All episodes must have the same length!"
        for buff, ep in zip(self.replay_buffers, episodes):
            for trans in ep:
                buff.append(
                    trans,
                    required_attrs=(
                        "state",
                        "action",
                        "next_state",
                        "reward",
                        "terminal",
                    ),
                )

    def update(
        self,
        update_value=True,
        update_policy=True,
        update_target=True,
        concatenate_samples=True,
    ):
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
        sample_indexes = [
            [randint(0, buffer_length - 1) for _ in range(batch_size)]
            for __ in range(self.ensemble_size)
        ]

        sample_methods = [
            self._create_sample_method(indexes) for indexes in sample_indexes
        ]

        # Now sample from buffer for each sub-policy in the ensemble.
        # To reduce memory usage, for each sub-policy "i" of each actor,
        # the same sample "i" will be used for training.

        # Tensors in the sampled batch will be moved to shared memory.

        # size: [ensemble size, num of actors]
        batches = []
        next_actions_t = []
        for e_idx in range(self.ensemble_size):
            ensemble_batch = []
            for a_idx in range(len(self.actors)):
                batch_size_, batch = self.replay_buffers[a_idx].sample_batch(
                    self.batch_size,
                    concatenate_samples,
                    sample_method=sample_methods[e_idx],
                    sample_attrs=[
                        "state",
                        "action",
                        "reward",
                        "next_state",
                        "terminal",
                        "*",
                    ],
                )
                ensemble_batch.append(batch)
                assert batch_size_ == batch_size

            batches.append(ensemble_batch)
            next_actions_t.append(
                [
                    self.action_transform_function(act)
                    for act in self.act(
                        [batch[3] for batch in ensemble_batch], target=True
                    )
                ]
            )

        if self.pool_type == "process":
            batches = self._move_to_shared_mem(batches)
            next_actions_t = self._move_to_shared_mem(next_actions_t)

        args = []
        self._update_counter += 1
        for e_idx in range(self.ensemble_size):
            for a_idx in range(len(self.actors)):
                args.append(
                    (
                        batch_size,
                        batches,
                        next_actions_t,
                        a_idx,
                        e_idx,
                        self.actors,
                        self.actor_targets,
                        self.critics,
                        self.critic_targets,
                        self.critic_visible_actors,
                        self.actor_optims,
                        self.critic_optims,
                        update_value,
                        update_policy,
                        update_target,
                        self.action_transform_function,
                        self.action_concat_function,
                        self.state_concat_function,
                        self.reward_function,
                        self.criterion,
                        self.discount,
                        self.update_rate,
                        self.update_steps,
                        self._update_counter,
                        self.grad_max,
                        self.visualize and not self.has_visualized,
                        self.visualize_dir,
                        self._backward,
                    )
                )
        all_loss = self.pool.starmap(self._update_sub_policy, args)
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
        super().load(model_dir, network_map, version)
        with t.no_grad():
            self.pool.starmap(
                hard_update,
                zip(
                    itertools.chain(*self.actors), itertools.chain(*self.actor_targets)
                ),
            )
            self.pool.starmap(hard_update, zip(self.critics, self.critic_targets))

    @staticmethod
    def _no_grad_safe_call(model, *named_args):
        with t.no_grad():
            result = safe_call(model, *named_args)
            return result

    @staticmethod
    def _jit_safe_call(model, *named_args):
        if not hasattr(model, "input_device") or not hasattr(model, "output_device"):
            # try to automatically determine the input & output
            # device of the model
            mt = type(model)
            device = determine_device(model)
            if len(device) > 1:
                raise RuntimeError(
                    f"""\
                    Failed to automatically determine i/o device of your model: {mt}
                    Detected multiple devices: {device}
    
                    You need to manually specify i/o device of your model.
    
                    Either Wrap your model of type nn.Module with one of:
                    1. static_module_wrapper from machin.model.nets.base
                    2. dynamic_module_wrapper from machin.model.nets.base 
                    
                    Or construct your own module & model with: 
                    NeuralNetworkModule from machin.model.nets.base"""
                )
            else:
                # assume that i/o devices are the same as parameter device
                # print a warning
                default_logger.warning(
                    f"""\
                    You have not specified the i/o device of your model {mt}
                    Automatically determined and set to: {device[0]}
    
                    The framework is not responsible for any un-matching device issues 
                    caused by this operation."""
                )
                model = static_module_wrapper(model, device[0], device[0])
        input_device = model.input_device
        # set in __init__
        args = model.arg_spec.args[1:] + model.arg_spec.kwonlyargs
        if model.arg_spec.defaults is not None:
            args_with_defaults = args[-len(model.arg_spec.defaults) :]
        else:
            args_with_defaults = []
        required_args = (
            set(args)
            - set(args_with_defaults)
            - set(
                model.arg_spec.kwonlydefaults.keys()
                if model.arg_spec.kwonlydefaults is not None
                else []
            )
        )
        model_type = model.model_type
        # t.jit._fork does not support keyword args
        # fill arguments in by their positions.
        args_list = [None for _ in args]
        args_filled = [False for _ in args]

        for na in named_args:
            for k, v in na.items():
                if k in args:
                    if k not in args:
                        pass
                    args_filled[args.index(k)] = True
                    if t.is_tensor(v):
                        args_list[args.index(k)] = v.to(input_device)
                    else:
                        args_list[args.index(k)] = v

        if not all(args_filled):
            not_filled = [arg for filled, arg in zip(args_filled, args) if not filled]
            req_not_filled = set(not_filled).intersection(required_args)
            if len(req_not_filled) > 0:
                raise RuntimeError(
                    f"""\
                    Required arguments of the forward function of Model {model_type} 
                    is {required_args}, missing required arguments: {req_not_filled}
        
                    Check your storage functions.
                    """
                )

        return t.jit._fork(model, *args_list)

    @staticmethod
    def _update_sub_policy(
        batch_size,
        batches,
        next_actions_t,
        actor_index,
        policy_index,
        actors,
        actor_targets,
        critics,
        critic_targets,
        critic_visible_actors,
        actor_optims,
        critic_optims,
        update_value,
        update_policy,
        update_target,
        atf,
        acf,
        scf,
        rf,
        criterion,
        discount,
        update_rate,
        update_steps,
        update_counter,
        grad_max,
        visualize,
        visualize_dir,
        backward_func,
    ):
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
        ensemble_n_act_t = next_actions_t[policy_index]
        visible_actors = critic_visible_actors[actor_index]

        actors[actor_index][policy_index].train()
        critics[actor_index].train()

        with t.no_grad():
            # only select visible actors
            all_next_actions_t = [
                ensemble_n_act_t[a_idx]
                if a_idx != actor_index
                else atf(
                    safe_call(
                        actor_targets[actor_index][policy_index],
                        ensemble_batch[a_idx][3],
                    )[0],
                    ensemble_batch[a_idx][5],
                )
                for a_idx in visible_actors
            ]
            all_next_actions_t = acf(all_next_actions_t)

            all_actions = [ensemble_batch[a_idx][1] for a_idx in visible_actors]
            all_actions = acf(all_actions)

            all_next_states = [ensemble_batch[a_idx][3] for a_idx in visible_actors]
            all_next_states = scf(all_next_states)

            all_states = [ensemble_batch[a_idx][0] for a_idx in visible_actors]
            all_states = scf(all_states)

        # Update critic network first
        # Generate target value using target critic.
        with t.no_grad():
            reward = ensemble_batch[actor_index][2]
            terminal = ensemble_batch[actor_index][4]
            next_value = safe_call(
                critic_targets[actor_index], all_next_states, all_next_actions_t
            )[0]
            next_value = next_value.view(batch_size, -1)
            y_i = rf(
                reward, discount, next_value, terminal, ensemble_batch[actor_index][5]
            )

        cur_value = safe_call(critics[actor_index], all_states, all_actions)[0]
        value_loss = criterion(cur_value, y_i.to(cur_value.device))

        if visualize:
            # only invoked if not running by pool
            MADDPG._visualize(value_loss, f"critic_{actor_index}", visualize_dir)

        if update_value:
            critics[actor_index].zero_grad()
            backward_func(value_loss)
            nn.utils.clip_grad_norm_(critics[actor_index].parameters(), grad_max)
            critic_optims[actor_index].step()

        # Update actor network
        all_actions = [ensemble_batch[a_idx][1] for a_idx in visible_actors]
        # find the actor index in the view range of critic
        # Eg: there are 4 actors in total: a_0, a_1, a_2, a_3
        # critic may have access to actor a_1 and a_2
        # then:
        #     visible_actors.index(a_1) = 0
        #     visible_actors.index(a_2) = 1
        # visible_actors.index returns the (critic-)local position of actor
        # in the view range of its corresponding critic.
        all_actions[visible_actors.index(actor_index)] = atf(
            safe_call(
                actors[actor_index][policy_index], ensemble_batch[actor_index][3]
            )[0],
            ensemble_batch[actor_index][5],
        )
        all_actions = acf(all_actions)

        act_value = safe_call(critics[actor_index], all_states, all_actions)[0]

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()

        if visualize:
            # only invoked if not running by pool
            MADDPG._visualize(
                act_policy_loss, f"actor_{actor_index}_{policy_index}", visualize_dir,
            )

        if update_policy:
            actors[actor_index][policy_index].zero_grad()
            backward_func(act_policy_loss)
            nn.utils.clip_grad_norm_(
                actors[actor_index][policy_index].parameters(), grad_max
            )
            actor_optims[actor_index][policy_index].step()

        # Update target networks
        if update_target:
            if update_rate is not None:
                soft_update(
                    actor_targets[actor_index][policy_index],
                    actors[actor_index][policy_index],
                    update_rate,
                )
                soft_update(
                    critic_targets[actor_index], critics[actor_index], update_rate
                )
            else:
                if update_counter % update_steps == 0:
                    hard_update(
                        actor_targets[actor_index][policy_index],
                        actors[actor_index][policy_index],
                    )
                    hard_update(critic_targets[actor_index], critics[actor_index])

        actors[actor_index][policy_index].eval()
        critics[actor_index].eval()
        return -act_policy_loss.item(), value_loss.item()

    @staticmethod
    def _visualize(final_tensor, name, directory):
        g = make_dot(final_tensor)
        g.render(
            filename=name, directory=directory, view=False, cleanup=False, quiet=True
        )

    @staticmethod
    def _move_to_shared_mem(obj):
        if t.is_tensor(obj):
            obj = obj.detach()
            obj.share_memory_()
            return obj
        elif isinstance(obj, list):
            for idx, sub_obj in enumerate(obj):
                obj[idx] = MADDPG._move_to_shared_mem(sub_obj)
            return obj
        elif isinstance(obj, tuple):
            obj = list(obj)
            for idx, sub_obj in enumerate(obj):
                obj[idx] = MADDPG._move_to_shared_mem(sub_obj)
            return tuple(obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = MADDPG._move_to_shared_mem(v)
            return obj

    @staticmethod
    def _check_parameters_device(models):
        devices = set()
        for model in models:
            for k, v in model.named_parameters():
                devices.add(v.device.type)
                if len(devices) > 1:
                    raise RuntimeError(
                        "All of your models should either"
                        "locate on GPUs or on your CPU!"
                    )
        return list(devices)[0]

    @staticmethod
    def _create_sample_method(indexes):
        def sample_method(buffer, _len):
            nonlocal indexes
            batch = [buffer[i] for i in indexes if i < len(buffer)]
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
            all_actions[k] = t.cat([act[k].cpu() for act in actions], dim=1)
        return all_actions

    @staticmethod
    def state_concat_function(states: List[Dict], *_):
        # Assume an atom state is [batch_size, state_dim]
        # concatenate states in the second dimension.
        # becomes [batch_size, actor_num * state_dim]
        keys = states[0].keys()
        all_states = {}
        for k in keys:
            all_states[k] = t.cat([st[k].cpu() for st in states], dim=1)
        return all_states

    @staticmethod
    def reward_function(reward, discount, next_value, terminal, *_):
        next_value = next_value.to(reward.device)
        terminal = terminal.to(reward.device)
        return reward + discount * ~terminal * next_value

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        default_values = {
            "models": [["Actor"], ["Actor"], ["Critic"], ["Critic"]],
            "model_args": ([()], [()], [()], [()]),
            "model_kwargs": ([{}], [{}], [{}], [{}]),
            "optimizer": "Adam",
            "criterion": "MSELoss",
            "criterion_args": (),
            "criterion_kwargs": {},
            "critic_visible_actors": None,
            "sub_policy_num": 0,
            "lr_scheduler": None,
            "lr_scheduler_args": None,
            "lr_scheduler_kwargs": None,
            "batch_size": 100,
            "update_rate": 0.001,
            "update_steps": None,
            "actor_learning_rate": 0.0005,
            "critic_learning_rate": 0.001,
            "discount": 0.99,
            "gradient_max": np.inf,
            "replay_size": 500000,
            "replay_device": "cpu",
            "replay_buffer": None,
            "visualize": False,
            "visualize_dir": "",
            "use_jit": True,
            "pool_type": "thread",
            "pool_size": None,
        }
        config = deepcopy(config)
        config["frame"] = "MADDPG"
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
        f_config = deepcopy(config["frame_config"])
        all_models = []
        for models, model_args, model_kwargs in zip(
            f_config["models"], f_config["model_args"], f_config["model_kwargs"]
        ):
            models = assert_and_get_valid_models(models)
            models = [
                m(*arg, **kwarg).to(model_device)
                for m, arg, kwarg in zip(models, model_args, model_kwargs)
            ]
            all_models.append(models)
        optimizer = assert_and_get_valid_optimizer(f_config["optimizer"])
        criterion = assert_and_get_valid_criterion(f_config["criterion"])(
            *f_config["criterion_args"], **f_config["criterion_kwargs"]
        )
        lr_scheduler = f_config["lr_scheduler"] and assert_and_get_valid_lr_scheduler(
            f_config["lr_scheduler"]
        )
        f_config["optimizer"] = optimizer
        f_config["criterion"] = criterion
        f_config["lr_scheduler"] = lr_scheduler
        frame = cls(*all_models, **f_config)
        return frame
