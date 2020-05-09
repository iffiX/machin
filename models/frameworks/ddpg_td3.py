from .ddpg import *


class DDPG_TD3(DDPG):
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 actor_target: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 critic_target: Union[NeuralNetworkModule, nn.Module],
                 critic2: Union[NeuralNetworkModule, nn.Module],
                 critic2_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=100,
                 update_rate=0.005,
                 discount=0.99,
                 replay_size=500000,
                 replay_device="cpu",
                 reward_func=None,
                 policy_noise_func=None,
                 action_trans_func=None):
        """
        Initialize DDPG_TD3 framework.
        """
        super(DDPG_TD3, self).__init__(actor, actor_target, critic, critic_target,
                                       optimizer, criterion, learning_rate,
                                       lr_scheduler, lr_scheduler_params,
                                       batch_size, update_rate, discount, replay_size,
                                       replay_device, reward_func, action_trans_func)
        self.critic2 = critic2
        self.critic2_target = critic2_target
        self.critic2_optim = optimizer(self.critic2.parameters(), lr=learning_rate)

        # Make sure target and online networks have the same weight
        with torch.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)
            hard_update(self.critic2, self.critic2_target)

        if lr_scheduler is not None:
            self.actor_lr_sch = lr_scheduler(self.actor_optim, *lr_scheduler_params[0])
            self.critic_lr_sch = lr_scheduler(self.critic_optim, *lr_scheduler_params[1])
            self.critic2_lr_sch = lr_scheduler(self.critic2_optim, *lr_scheduler_params[1])

        self.criterion = criterion

        self.reward_func = DDPG_TD3.bellman_function if reward_func is None else reward_func
        if policy_noise_func is None:
            raise RuntimeWarning("Policy noise function is None, no policy noise will be applied during update!")
        self.policy_noise_func = DDPG_TD3.policy_noise_function if policy_noise_func is None else policy_noise_func
        self.action_trans_func = DDPG.action_transform_function if action_trans_func is None else action_trans_func

        self.set_top(["actor", "critic", "critic2", "actor_target", "critic_target", "critic2_target"])
        self.set_restorable(["actor_target", "critic_target", "critic2_target"])

    def criticize2(self, state, action, use_target=False):
        """
        Use the second critic network to evaluate current value.

        Returns:
            Value evaluated by critic.
        Notes:
            State and action will be concatenated in dimension 1
        """
        if use_target:
            return safe_call(self.critic2_target, state, action)
        else:
            return safe_call(self.critic2, state, action)

    def set_policy_noise_function(self, pnf):
        """
        Set policy noise function
        """
        self.policy_noise_func = pnf

    def update(self, update_value=True, update_policy=True, update_targets=True, concatenate_samples=True):
        """
        Update network weights by sampling from replay buffer.

        Returns:
            (mean value of estimated policy value, value loss)
        """
        batch_size, (state, action, reward, next_state, terminal, *others) = \
            self.rpb.sample_batch(self.batch_size, concatenate_samples,
                                  sample_keys=["state", "action", "reward", "next_state", "terminal", "*"])

        # Update critic network first
        # Generate value reference :math: `y_i` using target actor and target critic
        with torch.no_grad():
            next_action = self.action_trans_func(self.policy_noise_func(self.act(next_state, True)),
                                                 next_state, *others)
            next_value = self.criticize(next_state, next_action, True)
            next_value2 = self.criticize2(next_state, next_action, True)
            next_value = torch.min(next_value, next_value2)
            next_value = next_value.view(batch_size, -1)
            y_i = self.reward_func(reward, self.discount, next_value, terminal, *others)

        cur_value = self.criticize(state, action)
        cur_value2 = self.criticize2(state, action)
        value_loss = self.criterion(cur_value, y_i.to(cur_value.device))
        value_loss2 = self.criterion(cur_value2, y_i.to(cur_value.device))

        if update_value:
            self.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()
            self.critic2.zero_grad()
            value_loss2.backward()
            self.critic2_optim.step()

        # Update actor network
        cur_action = self.action_trans_func(self.act(state), state, *others)
        act_value = self.criticize(state, cur_action)

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()

        if update_policy:
            self.actor.zero_grad()
            act_policy_loss.backward()
            self.actor_optim.step()

        # Update target networks
        if update_targets:
            soft_update(self.actor_target, self.actor, self.update_rate)
            soft_update(self.critic_target, self.critic, self.update_rate)
            soft_update(self.critic2_target, self.critic2, self.update_rate)

        # use .item() to prevent memory leakage
        return -act_policy_loss.item(), (value_loss.item() + value_loss2.item()) / 2

    @staticmethod
    def policy_noise_function(actions, *args):
        return actions

    def load(self, model_dir, network_map=None, version=-1):
        TorchFramework.load(model_dir, network_map, version)
        with torch.no_grad():
            hard_update(self.actor, self.actor_target)
            hard_update(self.critic, self.critic_target)
            hard_update(self.critic2, self.critic2_target)