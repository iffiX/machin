from .a2c import *
from utils.parallel import Pool, SimpleQueue, current_process


class A3C(A2C):
    def __init__(self,
                 actor: Union[NeuralNetworkModule, nn.Module],
                 critic: Union[NeuralNetworkModule, nn.Module],
                 optimizer,
                 criterion,
                 worker_num=2,
                 entropy_weight=None,
                 value_weight=0.5,
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
        Initialize A3C framework.
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
        super(A3C, self).__init__(actor, critic, optimizer, criterion,
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
        self.actor.share_memory()
        self.critic.share_memory()
        self.worker_pool = Pool(worker_num)
        self.worker_pool.enable_global_find(True)
        # For safety reasons, use full tensor copy rather than reference.
        # Usually in a3c, tensor copy between workers and the main process
        # is extremely rare.
        self.worker_pool.enable_copy_tensors(True)

        # send & recv are from the perspective of the parent process
        # ie. send to workers and recv from workers
        self.send_message_queue = SimpleQueue(self.worker_pool._ctx)
        self.recv_message_queue = SimpleQueue(self.worker_pool._ctx)

    def run_workers(self, worker_func):
        self.worker_pool.map(worker_func)

    def send(self, message):
        """
        Send additional message between parent and workers. Designed for logging purpose.
        """
        if current_process().name == 'MainProcess':
            self.send_message_queue.put(message)
        else:
            self.recv_message_queue.put(message)

    def recv(self):
        """
        Receive message from parent or workers. Designed for logging purpose.
        """
        if current_process().name == 'MainProcess':
            return self.recv_message_queue.get()
        else:
            return self.send_message_queue.get()