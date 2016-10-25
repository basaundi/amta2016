
from __future__ import print_function, division, absolute_import, unicode_literals
import pickle
import numpy  # TODO: Abstract this part
from . import zeano


class Parametric(object):
    def __init__(self, **kwargs):
        self.path = None
        if not hasattr(self, '_params'):
            self._params = set()
        if not hasattr(self, '_hyperparams'):
            self._hyperparams = {}

        for k, v in kwargs.items():
            self._hyperparams[k] = v

    @property
    def parameters(self):
        return {k: getattr(self, k) for k in self._params}

    def hyperparameter(self, name, default, help=None):
        if name not in self._hyperparams:
            self._hyperparams[name] = default
        setattr(self, name, self._hyperparams[name])

    def get_shared_by(self):
        if hasattr(self, "shared_by"):
            return self.shared_by
        if hasattr(self, "parent"):
            return self.parent.get_shared_by()
        return 1

    def parameter(self, name, init, override=False):
        assert name not in self._params or override, 'Parameter `{}` already declared.'.format(name)
        init = self._wrap_parameter(name, init)
        self._params.add(name)
        setattr(self, name, init)
        if isinstance(init, Parametric):
            init.parent = self
        elif hasattr(init, 'get_value'):
            init.parent = self
            init.get_shared_by = lambda: self.get_shared_by()
        return init

    @staticmethod
    def _wrap_parameter(name, init):
        if isinstance(init, numpy.ndarray):
            init = zeano.shared(init.astype(zeano.floatX), name, borrow=True)
        return init

    def save(self, path):
        if not self.path:
            self.path = path
        d = self.to_dict()
        with open(path, 'wb') as fd:
            pickle.dump(d, fd, protocol=2)

    # TODO: Rewrite
    def to_dict(self, done=None):
        if done is None:
            done = set()
        d = {k: getattr(self, k) for k in self._hyperparams if hasattr(self, k)}
        for k, v in self.parameters.items():
            if id(v) in done:
                continue
            done.add(id(v))
            if isinstance(v, Parametric):
                v = v.to_dict(done)
            elif hasattr(v, 'get_value'):
                v = v.get_value()
            d[k] = v
        return d

    def load(self, path):
        self.path = path
        with open(path, 'rb') as fd:
            d = pickle.load(fd)
        self.from_dict(d)

    # TODO: Rewrite
    def from_dict(self, d):
        for k, v in d.items():
            if hasattr(self, k) and isinstance(getattr(self, k), Parametric):
                getattr(self, k).from_dict(v)
            elif hasattr(self, k) and hasattr(getattr(self, k), 'get_value'):
                getattr(self, k).set_value(v)
            else:
                setattr(self, k, v)

    def flat_parameters(self, p=None):
        if not p:
            p = self.parameters
        ks, vs = [], []
        for k, v in p.items():
            if isinstance(v, Parametric):
                params = self.flat_parameters(v.parameters)
                for nk, nv in params.items():
                    ks.append("{}.{}".format(k, nk))
                    vs.append(nv)
            else:
                ks.append(k)
                vs.append(v)
        return dict(zip(ks, vs))

    def reset(self):
        # TODO: Reset childs.
        pass
