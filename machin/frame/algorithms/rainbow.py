from machin.frame.buffers.prioritized_buffer import PrioritizedBuffer

# pylint: disable=wildcard-import, unused-wildcard-import
from .dqn import *


class RAINBOW(DQN):
    """
    RAINBOW DQN framework.
    """

    def __init__(
        self,
        qnet: Union[NeuralNetworkModule, nn.Module],
        qnet_target: Union[NeuralNetworkModule, nn.Module],
        optimizer,
        value_min,
        value_max,
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple] = None,
        lr_scheduler_kwargs: Tuple[Dict] = None,
        batch_size: int = 100,
        epsilon_decay: float = 0.9999,
        update_rate: float = 0.001,
        update_steps: Union[int, None] = None,
        learning_rate: float = 0.001,
        discount: float = 0.99,
        gradient_max: float = np.inf,
        reward_future_steps: int = 3,
        replay_size: int = 500000,
        replay_device: Union[str, t.device] = "cpu",
        replay_buffer: Buffer = None,
        visualize: bool = False,
        visualize_dir: str = "",
        **__
    ):
        """
        RAINBOW framework is described in
        `this <https://arxiv.org/abs/1710.02298>`__ essay.

        Note:
            In the RAINBOW framework, the output shape of your q network
            must be ``[batch_size, action_num, atom_num]`` when given a
            state of shape ``[batch_size, action_dim]``. And the last
            dimension **must be soft-maxed**. Atom number is the number of
            segments of your q value domain.

        See Also:
            :class:`.DQN`

        Args:
            qnet: Q network module.
            qnet_target: Target Q network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            value_min: Minimum of value domain.
            value_max: Maximum of value domain.
            learning_rate: Learning rate of the optimizer, not compatible with
                ``lr_scheduler``.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            batch_size: Batch size used during training.
            epsilon_decay: Epsilon decay rate per acting with noise step.
                ``epsilon`` attribute is multiplied with this every time
                ``act_discrete_with_noise`` is called.
            update_rate: :math:`\\tau` used to update target networks.
                Target parameters are updated as:

                :math:`\\theta_t = \\theta * \\tau + \\theta_t * (1 - \\tau)`
            update_steps: Training step number used to update target networks.
            discount: :math:`\\gamma` used in the bellman function.
            reward_future_steps: Number of future steps to be considered when
                the framework calculates value from reward.
            replay_size: Replay buffer size. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer.
            mode: one of ``"vanilla", "fixed_target", "double"``.
            visualize: Whether visualize the network flow in the first pass.
        """
        super().__init__(
            qnet,
            qnet_target,
            optimizer,
            lambda: None,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            batch_size=batch_size,
            epsilon_decay=epsilon_decay,
            update_rate=update_rate,
            update_steps=update_steps,
            discount=discount,
            gradient_max=gradient_max,
            replay_size=replay_size,
            replay_device=replay_device,
            replay_buffer=(
                PrioritizedBuffer(replay_size, replay_device)
                if replay_buffer is None
                else replay_buffer
            ),
            visualize=visualize,
            visualize_dir=visualize_dir,
        )
        self.v_min = value_min
        self.v_max = value_max
        self.reward_future_steps = reward_future_steps

    def act_discrete(self, state: Dict[str, Any], use_target: bool = False, **__):
        # DOC INHERITED
        # q value distribution of each action
        # shape: [batch_size, action_num, atom_num]
        if use_target:
            q_dist, *others = safe_call(self.qnet_target, state)
        else:
            q_dist, *others = safe_call(self.qnet, state)

        atom_num = q_dist.shape[-1]

        # support vector, shape: [1, atom_num]
        q_dist_support = t.linspace(self.v_min, self.v_max, atom_num).view(1, -1)

        # q value of each action, shape: [batch_size, action_num]
        q_value = t.sum(q_dist_support.type_as(q_dist) * q_dist, dim=-1)

        result = t.argmax(q_value, dim=1).view(-1, 1)
        if len(others) == 0:
            return result
        else:
            return (result, *others)

    def act_discrete_with_noise(
        self,
        state: Dict[str, Any],
        use_target: bool = False,
        decay_epsilon: bool = True,
        **__
    ):
        # DOC INHERITED
        # q value distribution of each action
        # shape: [batch_size, action_num, atom_num]
        if use_target:
            q_dist, *others = safe_call(self.qnet_target, state)
        else:
            q_dist, *others = safe_call(self.qnet, state)

        atom_num = q_dist.shape[-1]

        # support vector, shape: [1, atom_num]
        q_dist_support = t.linspace(self.v_min, self.v_max, atom_num).view(1, -1)

        # q value of each action, shape: [batch_size, action_num]
        q_value = t.sum(q_dist_support.type_as(q_dist) * q_dist, dim=-1)

        action_dim = q_value.shape[1]
        result = t.argmax(q_value, dim=1).view(-1, 1)

        if t.rand([1]).item() < self.epsilon:
            result = t.randint(0, action_dim, [result.shape[0], 1])

        if decay_epsilon:
            self.epsilon *= self.epsilon_decay

        if len(others) == 0:
            return result
        else:
            return (result, *others)

    def store_transition(self, transition: Union[Transition, Dict]):
        """
        Add a transition sample to the replay buffer.

        Not suggested, since you will have to calculate "value"
        by yourself.
        """
        self.replay_buffer.append(
            transition,
            required_attrs=(
                "state",
                "action",
                "next_state",
                "reward",
                "value",
                "terminal",
            ),
        )

    def store_episode(self, episode: List[Union[Transition, Dict]]):
        """
        Add a full episode of transition samples to the replay buffer.

        "value" is automatically calculated.
        """
        episode[-1]["value"] = episode[-1]["reward"]

        # calculate (truncated) n step value for each transition
        for i in reversed(range(len(episode))):
            value_sum = 0
            # for (virtual) transitions beyond the terminal transition,
            # using "min" to ignore them is equivalent to setting their
            # rewards as zero
            for j in reversed(range(min(self.reward_future_steps, len(episode) - i))):
                value_sum = value_sum * self.discount + episode[i + j]["reward"]
            episode[i]["value"] = value_sum

        for trans in episode:
            self.replay_buffer.append(
                trans,
                required_attrs=(
                    "state",
                    "action",
                    "next_state",
                    "reward",
                    "value",
                    "terminal",
                ),
            )

    def update(
        self, update_value=True, update_target=True, concatenate_samples=True, **__
    ):
        # DOC INHERITED
        # pylint: disable=invalid-name
        self.qnet.train()
        (
            batch_size,
            (state, action, value, next_state, terminal, others),
            index,
            is_weight,
        ) = self.replay_buffer.sample_batch(
            self.batch_size,
            concatenate_samples,
            sample_attrs=["state", "action", "value", "next_state", "terminal", "*"],
            additional_concat_attrs=["value"],
        )

        # q_dist is the distribution of q values
        q_dist = self._criticize(state).cpu()
        atom_num = q_dist.shape[-1]

        action = (
            self.action_get_function(action).to(device="cpu", dtype=t.long).flatten()
        )
        # shape: [batch_size, atom_num]
        q_dist = q_dist[range(batch_size), action]

        # support vector, shape: [atom_num]
        q_dist_support = t.linspace(self.v_min, self.v_max, atom_num)

        with t.no_grad():
            target_next_q_dist = self._criticize(next_state, True).cpu()
            next_action = (
                self.act_discrete(next_state).flatten().to(device="cpu", dtype=t.long)
            )

            # shape: [batch_size, atom_num]
            target_next_q_dist = target_next_q_dist[range(batch_size), next_action]

            # shape: [1, atom_num]
            q_dist_support = q_dist_support.unsqueeze(dim=0)

            # T_z is the bellman update for atom z_j
            # shape: [batch_size, atom_num]
            T_z = self.reward_function(
                value.cpu(),
                self.discount ** self.reward_future_steps,
                q_dist_support,
                terminal.cpu(),
                others,
            )

            # 1e-6 is used to make sure that l != u when T_z == v_min or v_max
            T_z = T_z.clamp(self.v_min + 1e-6, self.v_max - 1e-6)

            # delta_z is the interval length of each atom
            delta_z = (self.v_max - self.v_min) / (atom_num - 1.0)

            # b is the normalized distance of T_z to v_min,
            # l and u are upper and lower atom indexes
            # b, l, u shape: [batch_size, atom_num]
            b = (T_z - self.v_min) / delta_z
            l, u = b.floor(), b.ceil()

            # idx shape: [batch_size * atom_num]
            # dist shape: [batch_size, atom_num]
            # weight shape: [batch_size * atom_num]
            l_idx, l_dist = l.long().view(-1), b - l
            u_idx, u_dist = u.long().view(-1), u - b
            l_weight = (u_dist * target_next_q_dist).view(-1)
            u_weight = (l_dist * target_next_q_dist).view(-1)

            # offset is used to perform row-wise index add, since we can only
            # perform index add on one dimension, we must flatten the whole
            # distribution and then add.
            offset = (
                t.arange(0, batch_size * atom_num, atom_num)
                .view(-1, 1)
                .expand(batch_size, atom_num)
                .flatten()
            )

            # distribute T_z probability to its nearest upper
            # and lower atom neighbors, using its distance to them.
            # shape: [batch_size * atom_num] -> [batch_size, atom_num]
            # Note: index_add_ on CUDA uses atomicAdd, will cause
            # rounding errors and be a source of noise.
            target_dist = t.zeros([batch_size * atom_num], dtype=l_weight.dtype)
            target_dist.index_add_(dim=0, index=l_idx + offset, source=l_weight)
            target_dist.index_add_(dim=0, index=u_idx + offset, source=u_weight)
            target_dist = target_dist.view(batch_size, atom_num)

        # target_dist is equivalent to y_i in original dqn
        # division in KL divergence is ignored because target_dist
        # is a constant? But this modification do prevents the 0/0 situation.

        # 1e-6 is used to improve numerical stability and prevent nan
        value_loss = -(target_dist * (q_dist + 1e-6).log())

        value_loss = value_loss.sum(dim=1)

        abs_error = (t.abs(value_loss) + 1e-6).flatten().detach().numpy()
        self.replay_buffer.update_priority(abs_error, index)

        value_loss = (value_loss * t.from_numpy(is_weight).view([batch_size, 1])).mean()

        if self.visualize:
            self.visualize_model(value_loss, "qnet", self.visualize_dir)

        if update_value:
            self.qnet.zero_grad()
            self._backward(value_loss)
            nn.utils.clip_grad_norm_(self.qnet.parameters(), self.grad_max)
            self.qnet_optim.step()

        # Update target Q network
        if update_target:
            if self.update_rate is not None:
                soft_update(self.qnet_target, self.qnet, self.update_rate)
            else:
                self._update_counter += 1
                if self._update_counter % self.update_steps == 0:
                    hard_update(self.qnet_target, self.qnet)

        self.qnet.eval()
        # use .item() to prevent memory leakage
        return value_loss.item()

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        config = DQN.generate_config(config)
        config["frame"] = "RAINBOW"
        config["frame_config"]["value_min"] = -1.0
        config["frame_config"]["value_max"] = 1.0
        config["frame_config"]["reward_future_steps"] = 3
        return config
