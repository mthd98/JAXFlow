from jaxflow.optimizers import BaseOptimizer
import optax

class Adam(BaseOptimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        super().__init__(learning_rate, beta1=beta1, beta2=beta2, eps=eps, **kwargs)

    def _create_optax_optimizer(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        return optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2, eps=eps)

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        })
        return config
