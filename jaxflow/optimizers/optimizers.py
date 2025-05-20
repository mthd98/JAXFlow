from .optimizer import BaseOptimizer
import optax


class Adam(BaseOptimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        super().__init__(learning_rate, **kwargs)

    def _create_optax_transform(self, learning_rate, **config):
        return optax.adam(learning_rate=learning_rate,
                          b1=self.beta1,
                          b2=self.beta2,
                          eps=self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'beta1': self.beta1, 'beta2': self.beta2, 'eps': self.eps})
        return cfg
      



class SGD(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float,
        momentum: float = 0.0,
        nesterov: bool = False,
        **kwargs,
    ):
        """
        Stochastic Gradient Descent optimizer with optional momentum.

        Args:
          learning_rate: The step size to use.
          momentum: Momentum coefficient (0 means vanilla SGD).
          nesterov: Whether to use Nesterov momentum.
          **kwargs: Passed to BaseOptimizer (weight_decay, clipnorm, etc.).
        """
        self.momentum = momentum
        self.nesterov = nesterov
        super().__init__(learning_rate, **kwargs)

    def _create_optax_transform(self, learning_rate: float, **config) -> optax.GradientTransformation:
        """Constructs the Optax SGD transform with momentum/nesterov."""
        return optax.sgd(
            learning_rate=learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'momentum': self.momentum,
            'nesterov': self.nesterov
        })
        return cfg
    

class RMSProp(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float,
        decay: float = 0.9,
        eps: float = 1e-8,
        momentum: float = 0.0,
        centered: bool = False,
        **kwargs,
    ):
        """
        RMSProp optimizer with optional momentum and centering.

        Args:
          learning_rate: step size.
          decay: smoothing constant (often called rho).
          eps: term added to the denominator to improve numerical stability.
          momentum: momentum coefficient (0 = no momentum).
          centered: if True, compute a “centered” RMSProp that normalizes by 
                    estimated variance (instead of uncentered second moment).
          **kwargs: passed through to BaseOptimizer (weight_decay, clipnorm, etc.).
        """
        self.decay = decay
        self.eps = eps
        self.momentum = momentum
        self.centered = centered
        super().__init__(learning_rate, **kwargs)

    def _create_optax_transform(
        self, learning_rate: float, **config
    ) -> optax.GradientTransformation:
        return optax.rmsprop(
            learning_rate=learning_rate,
            decay=self.decay,
            eps=self.eps,
            momentum=self.momentum,
            centered=self.centered
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'decay': self.decay,
            'eps': self.eps,
            'momentum': self.momentum,
            'centered': self.centered
        })
        return cfg 


class AdamW(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        **kwargs,
    ):
        """
        AdamW optimizer (Adam + decoupled weight decay).

        Args:
          learning_rate: step size.
          weight_decay: decoupled weight‐decay coefficient.
          beta1: exponential decay rate for first moment.
          beta2: exponential decay rate for second moment.
          eps: term for numerical stability.
          **kwargs: forwarded into BaseOptimizer (clipnorm, loss_scale, accumulate_steps, use_ema, etc.).
        """
        # store hyperparams
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.adamw_weight_decay = weight_decay

        # disable BaseOptimizer's own add_decayed_weights (we'll use optax.adamw)
        super().__init__(learning_rate, weight_decay=0.0, **kwargs)

    def _create_optax_transform(
        self, learning_rate: float, **config
    ) -> optax.GradientTransformation:
        # optax.adamw applies decoupled weight decay internally
        return optax.adamw(
            learning_rate=learning_rate,
            b1=self.beta1,
            b2=self.beta2,
            eps=self.eps,
            weight_decay=self.adamw_weight_decay,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.adamw_weight_decay,
        })
        return cfg
    

class Adamax(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        **kwargs,
    ):
        """
        AdaMax optimizer (Adam variant using the infinity norm).

        Args:
          learning_rate: step size.
          beta1: exponential decay rate for the 1st moment estimates.
          beta2: exponential decay rate for the ∞-norm moment estimates.
          eps: term added for numerical stability.
          **kwargs: passed through to BaseOptimizer (weight_decay, clipnorm, etc.).
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        super().__init__(learning_rate, **kwargs)

    def _create_optax_transform(
        self, learning_rate: float, **config
    ) -> optax.GradientTransformation:
        # optax.adamax implements the ∞-norm variant of Adam :contentReference[oaicite:0]{index=0}
        return optax.adamax(
            learning_rate=learning_rate,
            b1=self.beta1,
            b2=self.beta2,
            eps=self.eps
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
        })
        return cfg
    

class AdaDelta(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float,
        rho: float = 0.95,
        eps: float = 1e-6,
        **kwargs,
    ):
        """
        AdaDelta optimizer.

        Args:
          learning_rate: step size (often set to 1.0 for pure AdaDelta, but tunable).
          rho: decay rate for the moving averages (smoothing constant).
          eps: term added for numerical stability.
          **kwargs: passed through to BaseOptimizer (weight_decay, clipnorm, etc.).
        """
        self.rho = rho
        self.eps = eps
        super().__init__(learning_rate, **kwargs)

    def _create_optax_transform(
        self, learning_rate: float, **config
    ) -> optax.GradientTransformation:
        # optax.adadelta implements the AdaDelta update rule
        return optax.adadelta(
            learning_rate=learning_rate,
            rho=self.rho,
            eps=self.eps
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'rho': self.rho,
            'eps': self.eps,
        })
        return cfg
