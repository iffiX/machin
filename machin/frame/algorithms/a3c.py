from .a2c import *
from utils.parallel import Pool, SimpleQueue, current_process


class A3C(A2C):
    """
    A3C framework.
    """
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
        # DOC INHERITED
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