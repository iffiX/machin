from .a2c import *


class PPO(A2C):
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 entropy_weight=None,
                 value_weight=0.5,
                 surrogate_loss_clip=0.2,
                 gradient_max=np.inf,
                 gae_lambda=1.0,
                 learning_rate=0.001,
                 lr_scheduler=None,
                 lr_scheduler_params=None,
                 update_times=50,
                 discount=0.99,
                 replay_size=5000,
                 replay_device="cpu"):
        """
        Initialize PPO framework.
        Note: when given a state, (and an optional action) actor must at least return two
        values:
        1. Action
            For contiguous environments, action must be of shape [batch_size, action_dim]
            and clamped to environment limits.
            For discreet environments, action must be of shape [batch_size, action_dim],
            it could be a categorical encoded integer, or a one hot vector.

            Actions are given by samples during training in PPO framework. When actor is
            given a batch of actions and states, it must evaluate the states, and return
            the log likelihood of the given actions instead of re-sampling actions.

        2. Log likelihood of action (action probability)
            For contiguous environments, action's are not directly output by your actor,
            otherwise it would be rather inconvenient to generate this value, instead, your
            actor network should output parameters for a certain distribution (eg: normal)
            and then be drawn from it.

            For discreet environments, action probability is the one of the values in your
            one-hot vector. It is recommended to sample from torch.distribution.Categorical,
            instead of sampling by yourself.

            Action probability must be differentiable, actor will receive its gradient from
            the gradient of action probability.

        The third entropy value is optional:
        3. Entropy of action distribution (Optional)
            Entropy is usually calculated using dist.entropy(), it will be considered if you
            have specified the entropy_weight argument.

            An example of your actor in contiguous environments::

                class ActorNet(nn.Module):
                    def __init__(self):
                        super(ActorNet, self).__init__()
                        self.fc = nn.Linear(3, 100)
                        self.mu_head = nn.Linear(100, 1)
                        self.sigma_head = nn.Linear(100, 1)

                    def forward(self, state, action=None):
                        x = t.relu(self.fc(state))
                        mu = 2.0 * t.tanh(self.mu_head(x))
                        sigma = F.softplus(self.sigma_head(x))
                        dist = Normal(mu, sigma)
                        action = action if action is not None else dist.sample()
                        action_log_prob = dist.log_prob(action)
                        action_entropy = dist.entropy()
                        action = action.clamp(-2.0, 2.0)
                        return action.detach(), action_log_prob, action_entropy

        """
        super(PPO, self).__init__(actor, critic, optimizer, criterion,
                                  entropy_weight=entropy_weight,
                                  value_weight=value_weight,
                                  gradient_max=gradient_max,
                                  gae_lambda=gae_lambda,
                                  learning_rate=learning_rate,
                                  lr_scheduler=lr_scheduler,
                                  lr_scheduler_params=lr_scheduler_params,
                                  update_times=update_times,
                                  discount=discount,
                                  replay_size=replay_size,
                                  replay_device=replay_device)
        self.surr_clip = surrogate_loss_clip

    def update(self, update_value=True, update_policy=True, concatenate_samples=True):
        sum_act_policy_loss = 0
        sum_value_loss = 0

        # sample a batch
        batch_size, (state, action, reward, next_state, terminal,
                     action_log_prob, target_value, advantage, *others) = \
            self.rpb.sample_batch(-1,
                                  sample_method="all",
                                  concatenate=concatenate_samples,
                                  sample_keys=["state", "action", "reward", "next_state", "terminal",
                                               "action_log_prob", "value", "gae", "*"],
                                  additional_concat_keys=["action_log_prob", "value", "gae"])

        # normalize target value
        target_value = (target_value - target_value.mean()) / (target_value.std() + 1e-5)

        # Infer original action log probability
        __, action_log_prob, *_ = self.eval_act(state, action)
        action_log_prob = action_log_prob.view(batch_size, 1)

        for i in range(self.update_times):
            value = self.criticize(state)

            if self.entropy_weight is not None:
                __, new_action_log_prob, new_action_entropy, *_ = self.eval_act(state, action)

            else:
                __, new_action_log_prob, *_ = self.eval_act(state, action)

            new_action_log_prob = new_action_log_prob.view(batch_size, 1)

            # calculate surrogate loss
            sim_ratio = t.exp(new_action_log_prob - action_log_prob).detach()
            advantage = advantage.to(sim_ratio.device)
            surr_loss_1 = sim_ratio * advantage
            surr_loss_2 = t.clamp(sim_ratio, 1 - self.surr_clip, 1 + self.surr_clip) * advantage

            # calculate policy loss using surrogate loss
            act_policy_loss = -t.min(surr_loss_1, surr_loss_2)

            if self.entropy_weight is not None:
                act_policy_loss += self.entropy_weight * new_action_entropy.mean()

            act_policy_loss = act_policy_loss.mean()

            value_loss = self.criterion(target_value.to(value.device), value) * self.value_weight

            # Update actor network
            if update_policy:
                self.actor.zero_grad()
                act_policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_max)
                self.actor_optim.step()
                sum_act_policy_loss += act_policy_loss.item()

            # Update critic network
            if update_value:
                self.critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_max)
                self.critic_optim.step()
                sum_value_loss += value_loss.item()

        self.rpb.clear()
        return -sum_act_policy_loss / self.update_times, sum_value_loss / self.update_times
