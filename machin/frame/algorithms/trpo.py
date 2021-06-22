from .a2c import *
from .utils import safe_return
from machin.utils.logging import default_logger

# Implementation Reference: https://github.com/Khrylx/PyTorch-RL
# Implementation Reference: https://github.com/openai/spinningup


class TRPO(A2C):
    """
    TRPO framework.
    See `Trust Region Policy Optimization <https://arxiv.org/pdf/1502.05477.pdf>`_.
    """

    def __init__(
        self,
        actor: Union[NeuralNetworkModule, nn.Module],
        critic: Union[NeuralNetworkModule, nn.Module],
        optimizer: Callable,
        criterion: Callable,
        *_,
        lr_scheduler: Callable = None,
        lr_scheduler_args: Tuple[Tuple, Tuple] = (),
        lr_scheduler_kwargs: Tuple[Dict, Dict] = (),
        batch_size: int = 100,
        critic_update_times: int = 10,
        actor_learning_rate: float = 0.003,
        critic_learning_rate: float = 0.001,
        entropy_weight: float = None,
        value_weight: float = 0.5,
        gradient_max: float = np.inf,
        gae_lambda: float = 1.0,
        discount: float = 0.99,
        normalize_advantage: bool = True,
        kl_max_delta: float = 0.01,
        damping: float = 0.1,
        line_search_backtracks: int = 10,
        conjugate_eps: float = 1e-8,
        conjugate_iterations: int = 10,
        conjugate_res_threshold: float = 1e-10,
        hv_mode: str = "fim",
        replay_size: int = 500000,
        replay_device: Union[str, t.device] = "cpu",
        replay_buffer: Buffer = None,
        visualize: bool = False,
        visualize_dir: str = "",
        **__
    ):
        """
        See Also:
            :class:`.A2C`

        Important:
            TRPO requires a slightly different actor model compared to other
            stochastic policy algorithms such as A2C, PPO, etc.

            When given a state, and an optional action actor must
            at least return two values, these two values are the same
            as what A2C and PPO requires, for more information please refer to
            :class:`.A2C`.

            **1. Action**
            **2. Log likelihood of action (action probability)**

            The model must have another three methods:

            1. ``get_kl(self, states: Any, ...)``, returns kl divergence of the model
               when given a batch of state inputs. kl divergence is computed by:

               :math:`D_{KL}(\pi, \pi_k)= \sum(\pi_k)\log\frac{\pi_k}{\pi}`

               Where :math:`\pi_k = \pi` is the current policy model at iteration k,
               since parameter :math:`\theta_k = \theta`, you should detach
               :math`$\theta_k$` in the computation and make it fixed.

            2. ``compare_kl(self, params: t.Tensor, states: Any, ...)``, returns kl
               divergence between model with given params and model with current params,
               given params are flat.

            3. ``get_fim(self, states: Any, ...)``, returns the Fisher
               information matrix on mean parameter :math:`\mu` of the model when
               given a batch of state inputs.

               You can refer to this `article <https://www.ii.pwr.edu.pl/~tomczak/
PDF/[JMT]Fisher_inf.pdf>` on how to compute this matrix, note we only need fisher
               information matrix on the mean parameter.

               Since it's a diagonal matrix, we only need to return diagonal elements
               to fully represent it.

            Two base class for the discrete case and continuous case (which samples
            from a diagonal normal distribution) with this additional method are
            provided in :module:`machin.model.algorithms.trpo`, you **must** extend on
            these two base classes to create your model.

        Args:
            actor: Actor network module.
            critic: Critic network module.
            optimizer: Optimizer used to optimize ``actor`` and ``critic``.
            criterion: Criterion used to evaluate the value loss.
            lr_scheduler: Learning rate scheduler of ``optimizer``.
            lr_scheduler_args: Arguments of the learning rate scheduler.
            lr_scheduler_kwargs: Keyword arguments of the learning
                rate scheduler.
            batch_size: Batch size used only during training of critic. Actor train
                on the whole buffer.
            critic_update_times: Times to update critic in ``update()``.
            actor_learning_rate: Learning rate of the actor optimizer,
                not compatible with ``lr_scheduler``.
            critic_learning_rate: Learning rate of the critic optimizer,
                not compatible with ``lr_scheduler``.
            entropy_weight: Weight of entropy in your loss function, a positive
                entropy weight will minimize entropy, while a negative one will
                maximize entropy.
            value_weight: Weight of critic value loss.
            gradient_max: Maximum gradient.
            gae_lambda: :math:`\\lambda` used in generalized advantage
                estimation.
            discount: :math:`\\gamma` used in the bellman function.
            normalize_advantage: Whether to normalize sampled advantage values in
                the batch.
            kl_max_delta: Maximum delta allowed of the kl divergence between current
                model and the updated model.
            damping: Artifact for numerical stability, should be smallish.
                Adjusts Hessian-vector product calculation:

                :math:`Hv \\rightarrow (\\alpha I + H)v`

                where :math:`\\alpha` is the damping coefficient.
                Probably don't play with this hyperparameter.
                See `Conjugate gradient bundle adjustment <https://www1.maths.lth.se/
matematiklth/vision/publdb/reports/pdf/byrod-eccv-10.pdf>` equation 6.
            line_search_backtracks: Maximum number of times to try in the line search.
            conjugate_eps: A small constant used to prevent conjugate gradient from
                outputing nan value in the first iteration.
            conjugate_iterations: Maximum number of iterations of the conjugate gradient
                algorithm.
            conjugate_res_threshold: The threshold squared length of the residual
                vector.
            hv_mode: Which method to use to compute hessian vector product. One of
                "fim" or "direct", "fim" is faster and the default method.
            replay_size: Replay buffer size. Not compatible with
                ``replay_buffer``.
            replay_device: Device where the replay buffer locates on, Not
                compatible with ``replay_buffer``.
            replay_buffer: Custom replay buffer.
            visualize: Whether visualize the network flow in the first pass.
            visualize_dir: Visualized graph save directory.

        """
        super().__init__(
            actor,
            critic,
            optimizer,
            criterion,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            batch_size=batch_size,
            actor_update_times=1,
            critic_update_times=critic_update_times,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            entropy_weight=entropy_weight,
            value_weight=value_weight,
            gradient_max=gradient_max,
            gae_lambda=gae_lambda,
            discount=discount,
            normalize_advantage=normalize_advantage,
            replay_size=replay_size,
            replay_device=replay_device,
            replay_buffer=replay_buffer,
            visualize=visualize,
            visualize_dir=visualize_dir,
        )
        self.line_search_backtracks = line_search_backtracks
        self.kl_max_delta = kl_max_delta
        self.damping = damping
        self.conjugate_eps = conjugate_eps
        self.conjugate_iterations = conjugate_iterations
        self.conjugate_res_threshold = conjugate_res_threshold
        self.hv_mode = hv_mode

    def update(
        self, update_value=True, update_policy=True, concatenate_samples=True, **__
    ):
        # DOC INHERITED
        sum_value_loss = 0

        self.actor.train()
        self.critic.train()

        # sample a batch for actor training
        batch_size, (state, action, advantage) = self.replay_buffer.sample_batch(
            -1,
            sample_method="all",
            concatenate=concatenate_samples,
            sample_attrs=["state", "action", "gae"],
            additional_concat_attrs=["gae"],
        )

        # normalize advantage
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        # Train actor
        # define two closures needed by fvp functions
        ___, fixed_action_log_prob, *_ = self._eval_act(state, action)
        fixed_action_log_prob = fixed_action_log_prob.view(batch_size, 1).detach()
        fixed_params = self.get_flat_params(self.actor)

        def actor_loss_func():
            ____, action_log_prob, *_ = self._eval_act(state, action)
            action_log_prob = action_log_prob.view(batch_size, 1)
            action_loss = -advantage.to(action_log_prob.device) * t.exp(
                action_log_prob - fixed_action_log_prob
            )
            return action_loss.mean()

        def actor_kl_func():
            state["params"] = fixed_params
            return safe_return(safe_call(self.actor, state, method="compare_kl"))

        act_policy_loss = actor_loss_func()

        if self.visualize:
            self.visualize_model(act_policy_loss, "actor", self.visualize_dir)

        # Update actor network
        if update_policy:

            def fvp(v):
                if self.hv_mode == "fim":
                    return self._fvp_fim(state, v, self.damping)
                else:
                    return self._fvp_direct(state, v, self.damping)

            loss_grad = self.get_flat_grad(
                act_policy_loss, list(self.actor.parameters())
            ).detach()

            # usually 1e-15 is low enough
            if t.allclose(loss_grad, t.zeros_like(loss_grad), atol=1e-15):
                default_logger.warning(
                    "TRPO detects zero gradient, update step skipped."
                )
                return 0, 0

            step_dir = self._conjugate_gradients(
                fvp,
                -loss_grad,
                eps=self.conjugate_eps,
                iterations=self.conjugate_iterations,
                res_threshold=self.conjugate_res_threshold,
            )

            # Maximum step size mentioned in appendix C of the paper.
            beta = np.sqrt(2 * self.kl_max_delta / step_dir.dot(fvp(step_dir)).item())

            full_step = step_dir * beta
            if not self._line_search(
                self.actor, actor_loss_func, actor_kl_func, full_step, self.kl_max_delta
            ):
                default_logger.warning(
                    "Cannot find an update step to satisfy kl_max_delta, "
                    "consider increase line_search_backtracks"
                )

        for _ in range(self.critic_update_times):
            # sample a batch
            batch_size, (state, target_value) = self.replay_buffer.sample_batch(
                self.batch_size,
                sample_method="random_unique",
                concatenate=concatenate_samples,
                sample_attrs=["state", "value"],
                additional_concat_attrs=["value"],
            )
            # calculate value loss
            value = self._criticize(state)
            value_loss = (
                self.criterion(target_value.type_as(value), value) * self.value_weight
            )

            if self.visualize:
                self.visualize_model(value_loss, "critic", self.visualize_dir)

            # Update critic network
            if update_value:
                self.critic.zero_grad()
                self._backward(value_loss)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_max)
                self.critic_optim.step()
            sum_value_loss += value_loss.item()

        self.replay_buffer.clear()
        self.actor.eval()
        self.critic.eval()
        return (
            act_policy_loss,
            sum_value_loss / self.critic_update_times,
        )

    @staticmethod
    def _conjugate_gradients(Avp_f, b, eps, iterations, res_threshold):
        """
        The conjugate gradient method, which solves a linear system :math`Ax = b`.
        See `Conjugate gradient method \
    <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_.

        Args:
            Avp_f: A function which takes current basis vector :math`p` and
                returns :math`Ap`.
            b: RHS of :math`Ax = b`.
            iterations: Max number of iterations to run this algorithm
            res_threshold: The threshold squared length of the residual vector
                :math:`r_k`, where :math`k` is the iteration step.

        Returns:
            Solution of :math:`x`.
        """

        x = t.zeros(b.shape, dtype=b.dtype, device=b.device)
        r = b.clone()
        p = b.clone()
        r_dot_r = t.dot(r, r)
        for i in range(iterations):
            Avp = Avp_f(p)
            alpha = r_dot_r / (t.dot(p, Avp) + eps)
            x += alpha * p
            r -= alpha * Avp
            new_r_dot_r = t.dot(r, r)
            beta = new_r_dot_r / (r_dot_r + eps)
            p = r + beta * p
            r_dot_r = new_r_dot_r
            if r_dot_r < res_threshold:
                break
        return x

    @staticmethod
    def _line_search(
        model, loss_func, kl_func, full_step, kl_max_delta, max_backtracks=10
    ):
        flat_params = TRPO.get_flat_params(model)
        with t.no_grad():
            loss = loss_func().item()

            for fraction in [0.5 ** i for i in range(max_backtracks)]:
                new_params = flat_params + fraction * full_step
                TRPO.set_flat_params(model, new_params)
                new_loss = loss_func().item()
                improve = loss - new_loss

                # Note: some implementations like Khrylx/PyTorch-RL
                # use a method which compares delta of loss to:
                #
                #     expected_improve = -loss_grad.dot(full_step)
                #
                # and then compute a ratio, if ratio > 0.1, break out
                # of the iteration:
                #
                #     ratio = actual_improve / expected_improve
                #
                # since the meaning of this method is not clearly stated
                # anywhere, we choose to obey the implementation in the
                # paper and openai/spinningup, which checks kl range and
                # loss.
                if kl_func() <= kl_max_delta and improve > 0:
                    return True
            TRPO.set_flat_params(model, flat_params)
            return False

    def _fvp_direct(self, state: Dict[str, Any], vector: t.Tensor, damping: float):
        """
        The generic way to compute the Fisher-vector product mentioned in Appendix
        C.1 of the paper.

        Args:
            state: State dictionary to be fed to the actor.
            vector: The vector to multiply with the Hessian matrix.
            damping: Coefficient for numerical stability.


        Returns:
            Matrix product of :math:`Hv`
        """
        kl = safe_return(safe_call(self.actor, state, method="get_kl"))
        kl = kl.mean()

        grads = t.autograd.grad(kl, list(self.actor.parameters()), create_graph=True)
        flat_grad_kl = t.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * vector).sum()
        grads = t.autograd.grad(kl_v, list(self.actor.parameters()))
        flat_grad_grad_kl = t.cat(
            [grad.contiguous().view(-1) for grad in grads]
        ).detach()

        return flat_grad_grad_kl + vector * damping

    def _fvp_fim(self, state: Dict[str, Any], vector: t.Tensor, damping: float):
        """
        The more optimized way to compute the Fisher-vector product mentioned in
        Appendix C.1 of the paper.

        Please refer to `this blog <https://www.telesens.co/2018/06/09/efficien
tly-computing-the-fisher-vector-product-in-trpo/>`_ for more details

        Args:
            state: State dictionary to be fed to the actor.
            vector: The vector to multiply with the Hessian matrix.
            damping: Coefficient for numerical stability.

        Returns:
            Matrix product of :math:`Hv`
        """
        batch_size = next(st.shape[0] for st in state.values() if t.is_tensor(st))

        # M is the second derivative of the KL distance w.r.t. network output
        # (M*M diagonal matrix compressed into a M*1 vector)
        M, act_param = safe_call(self.actor, state, method="get_fim")

        # From now on we will use symbol `mu` as the action parameter of the
        # distribution, this symbol is used in equation 56. and 57. of the
        # paper
        mu = act_param.view(-1)

        # t is an arbitrary constant vector that does not depend on actor parameter
        # theta, we use t_ here since torch is imported as t
        t_ = t.ones(mu.shape, requires_grad=True, device=mu.device)
        mu_t = (mu * t_).sum()
        Jt = self.get_flat_grad(mu_t, list(self.actor.parameters()), create_graph=True)
        Jtv = (Jt * vector).sum()
        Jv = t.autograd.grad(Jtv, t_)[0]
        MJv = M * Jv.detach()
        mu_MJv = (MJv * mu).sum()
        JTMJv = self.get_flat_grad(mu_MJv, list(self.actor.parameters())).detach()
        JTMJv /= batch_size
        return JTMJv + vector * damping

    @staticmethod
    def get_flat_params(model: nn.Module):
        """
        Return flattened param tensor of shape [n] of input model.
        """
        params = []
        for param in model.parameters():
            params.append(param.view(-1))

        flat_params = t.cat(params)
        return flat_params

    @staticmethod
    def set_flat_params(model: nn.Module, flat_params: t.Tensor):
        """
        Set model parameters according to the flattened parameter tensor.
        """
        idx = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.shape)))
            param.data.copy_(flat_params[idx : idx + flat_size].view(param.shape))
            idx += flat_size

    @staticmethod
    def get_flat_grad(
        output: t.Tensor,
        parameters: List[nn.Parameter],
        retain_graph=False,
        create_graph=False,
    ):
        """
        Compute gradient w.r.t. parameters and returns a flattened gradient tensor.

        Note: use a list of parameters since it is hard to reset the iterator provided
        by calling model.parameters() after t.autograd.grad call.
        """
        if create_graph:
            retain_graph = True

        # allow unused parameters in graph since some parameter (like action log std)
        # may not receive gradient in the first or both passes.
        grads = t.autograd.grad(
            output,
            parameters,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True,
        )
        out_grads = []
        for g, p in zip(grads, parameters):
            if g is not None:
                out_grads.append(g.view(-1))
            else:
                out_grads.append(t.zeros_like(p).view(-1))
        grads = t.cat(out_grads)

        for param in parameters:
            param.grad = None
        return grads

    @classmethod
    def generate_config(cls, config: Union[Dict[str, Any], Config]):
        config = A2C.generate_config(config)
        config["frame"] = "TRPO"
        config["frame_config"]["kl_max_delta"] = 0.01
        config["frame_config"]["damping"] = 0.1
        config["frame_config"]["line_search_backtracks"] = 10
        config["frame_config"]["conjugate_eps"] = 1e-8
        config["frame_config"]["conjugate_iterations"] = 10
        config["frame_config"]["conjugate_res_threshold"] = 1e-10
        config["frame_config"]["hv_mode"] = "fim"
        return config
