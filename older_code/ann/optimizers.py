
from __future__ import print_function, division, absolute_import, unicode_literals
from collections import OrderedDict
from .parameters import Parametric
import numpy as np
from .zeano import sqrt, switch, function, grad, using_theano


class BaseOptimizer(Parametric):
    def __init__(self, model, **kwargs):
        super(BaseOptimizer, self).__init__(**kwargs)
        self.model = model
        self.hyperparameter('regularization_L1', 0.0, "L1 regularization term.")
        self.hyperparameter('regularization_L2', 0.0, "L2 regularization term.")
        self.hyperparameter('max_gradient_norm', 10, "Clip gradients to this norm.")
        self.hyperparameter('epsilon', 1e-6, "Epsilon for optimizer.")

    def regularize(self, cost, parameters):
        if self.regularization_L1 > 0.0:
            for param in parameters:
                cost += abs(param).sum() * self.regularization_L1
        if self.regularization_L2 > 0.0:
            for param in parameters:
                cost += (param ** 2).sum() * self.regularization_L2
        return cost

    def clip(self, gradients):
        clip_c = self.max_gradient_norm
        if clip_c > 0.:
            g2 = sum((g**2).sum() for g in gradients)
            gradients = [switch(g2 > clip_c ** 2, g / sqrt(g2) * clip_c, g) for g in gradients]
        return gradients

    def get_cost_with_update(self):
        # FIXME: Avoid `using_theano` here.
        if using_theano:
            parameters = OrderedDict(self.model.flat_parameters())
            cost = self.model.cost(*self.model.cost_arguments)
            cost = self.regularize(cost, parameters)
            grads = grad(cost=cost, wrt=parameters.values())
            grads = self.clip(grads)
            updates = self.updates(parameters, grads)
            cost_fn = function(self.model.cost_arguments, cost, updates=updates)
        else:
            # FIXME: Properly apply updates
            cost_fn = self.model.cost
        return cost_fn

    def updates(self, parameters, grads):
        raise NotImplementedError


class Adadelta(BaseOptimizer):
    def __init__(self, model, **kwargs):
        super(Adadelta, self).__init__(model, **kwargs)
        self.hyperparameter('rho', 0.95, "Rho for Adadelta optimizer.")
        parameters = self.model.flat_parameters()
        for name, param in parameters.items():
            self.parameter('{}_gradient_sq'.format(name), np.zeros([0] * param.ndim))
            self.parameter('{}_delta_sq'.format(name), np.zeros([0] * param.ndim))

    def reset(self):
        parameters = self.model.flat_parameters()
        for name, param in parameters.items():
            getattr(self, '{}_gradient_sq'.format(name)).set_value(param.get_value() * 0.0)
            getattr(self, '{}_delta_sq'.format(name)).set_value(param.get_value() * 0.0)

    def updates(self, parameters, grads):
        rho, eps = self.rho, self.epsilon
        updates = []
        for (name, param), gradient in zip(parameters.items(), grads):
            gradient_sq = getattr(self, '{}_gradient_sq'.format(name))
            delta_sq = getattr(self, '{}_delta_sq'.format(name))
            gradient_sq_new = rho * gradient_sq + (1 - rho) * (gradient**2)
            delta = (sqrt(delta_sq+eps) / sqrt(gradient_sq_new + eps)) * gradient
            delta_sq_new = rho * delta_sq + (1 - rho) * (delta**2)

            updates.append((gradient_sq, gradient_sq_new))
            updates.append((delta_sq, delta_sq_new))
            updates.append((param, param - delta))
        return updates
