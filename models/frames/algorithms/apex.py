from .dqn_per import *
from .ddpg_per import *
from ..buffers.prioritized_buffer_d import DistributedPrioritizedBuffer
from utils.parallel.server import SimplePushPullServer


class DQN_APEX(DQN_PER):
    def __init__(self,
                 qnet: Union[NeuralNetworkModule, nn.Module],
                 qnet_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 worker_group,
                 trainer_group,
                 pull_function=None,
                 push_function=None,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=100,
                 update_rate=1.0,
                 discount=0.99,
                 replay_size=50000000,
                 replay_device="cpu",
                 reward_func=None):
        super(DQN_APEX, self).__init__(qnet, qnet_target, optimizer, criterion,
                                       learning_rate=learning_rate,
                                       lr_scheduler=lr_scheduler,
                                       lr_scheduler_params=lr_scheduler_params,
                                       batch_size=batch_size,
                                       update_rate=update_rate,
                                       discount=discount,
                                       replay_size=replay_size,
                                       replay_device=replay_device,
                                       reward_func=reward_func)
        self.rpb = DistributedPrioritizedBuffer(buffer_size=replay_size,
                                                buffer_group=trainer_group)

        if push_function is None or pull_function is None:
            self.pp = SimplePushPullServer(trainer_group)
            self.pull_function = self.pp.pull
            self.push_function = self.pp.push
        else:
            self.pull_function = pull_function
            self.push_function = push_function

    def act_discreet(self, state, use_target=False):
        if use_target:
            self.pull_function(self.qnet_target, "qnet_target")
        else:
            self.pull_function(self.qnet, "qnet")
        return super(DQN_APEX, self).act_discreet(state, use_target)

    def act_discreet_with_noise(self, state, use_target=False):
        if use_target:
            self.pull_function(self.qnet_target, "qnet_target")
        else:
            self.pull_function(self.qnet, "qnet")
        return super(DQN_APEX, self).act_discreet_with_noise(state, use_target)

    def criticize(self, state, use_target=False):
        if use_target:
            self.pull_function(self.qnet_target, "qnet_target")
        else:
            self.pull_function(self.qnet, "qnet")
        return super(DQN_APEX, self).criticize(state, use_target)

    def update(self, update_value=True, update_target=True, concatenate_samples=True):
        result = super(DQN_APEX, self).update(update_value, update_target, concatenate_samples)
        if update_target:
            self.push_function(self.qnet_target, "qnet_target")
        if update_value:
            self.push_function(self.qnet, "qnet")
        return result


class DDPG_APEX(DDPG_PER):
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 actor_target: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 critic_target: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 worker_group,
                 trainer_group,
                 pull_function=None,
                 push_function=None,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 batch_size=100,
                 update_rate=0.005,
                 discount=0.99,
                 replay_size=500000,
                 replay_device="cpu",
                 reward_func=None,
                 action_trans_func=None):
        super(DDPG_APEX, self).__init__(actor, actor_target, critic, critic_target,
                                        optimizer, criterion,
                                        learning_rate=learning_rate,
                                        lr_scheduler=lr_scheduler,
                                        lr_scheduler_params=lr_scheduler_params,
                                        batch_size=batch_size,
                                        update_rate=update_rate,
                                        discount=discount,
                                        replay_size=replay_size,
                                        replay_device=replay_device,
                                        reward_func=reward_func,
                                        action_trans_func=action_trans_func)
        self.rpb = DistributedPrioritizedBuffer(buffer_size=replay_size,
                                                buffer_group=trainer_group)

        if push_function is None or pull_function is None:
            self.pp = SimplePushPullServer(trainer_group)
            self.pull_function = self.pp.pull
            self.push_function = self.pp.push
        else:
            self.pull_function = pull_function
            self.push_function = push_function

    def act_discreet(self, state, use_target=False):
        if use_target:
            self.pull_function(self.actor_target, "actor_target")
        else:
            self.pull_function(self.actor, "actor")
        return super(DDPG_APEX, self).act_discreet(state, use_target)

    def act_discreet_with_noise(self, state, use_target=False):
        if use_target:
            self.pull_function(self.actor_target, "actor_target")
        else:
            self.pull_function(self.actor, "actor")
        return super(DDPG_APEX, self).act_discreet_with_noise(state, use_target)

    def criticize(self, state, action, use_target=False):
        if use_target:
            self.pull_function(self.critic_target, "critic_target")
        else:
            self.pull_function(self.critic, "critic")
        return super(DDPG_APEX, self).criticize(state, action, use_target)

    def update(self, update_value=True, update_policy=True, update_target=True, concatenate_samples=True):
        result = super(DDPG_APEX, self).update(update_value, update_policy,
                                               update_target, concatenate_samples)
        if update_target:
            self.push_function(self.critic_target, "critic_target")
            self.push_function(self.actor_target, "actor_target")
        if update_value:
            self.push_function(self.critic, "critic")
            self.push_function(self.actor, "actor")
        return result