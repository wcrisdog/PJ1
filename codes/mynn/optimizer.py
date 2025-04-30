from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, beta=0.9):
        super().__init__(init_lr, model)
        self.beta = beta
        # create per-layer, per-parameter velocity slots
        self.velocity = {}
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocity[layer] = {
                    key: np.zeros_like(param)
                    for key, param in layer.params.items()
                }

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key, param in layer.params.items():
                # apply weight decay exactly as in SGD
                if layer.weight_decay:
                    param *= (1 - self.init_lr * layer.weight_decay_lambda)

                # fetch grad and previous velocity
                grad = layer.grads[key]
                v_prev = self.velocity[layer][key]

                # momentum update
                v_new = self.beta * v_prev - self.init_lr * grad
                self.velocity[layer][key] = v_new

                # apply update
                layer.params[key] += v_new