""" Abstract theano and numpy.
"""

__all__ = (
    'exp', 'tanh', 'zeros', 'dot', 'mean', 'concatenate', 'log',
    'zeros_like', 'arange', 'sqrt', 'tile', 'roll', 'sigmoid',
    'ones_like',

    'lvector', 'lmatrix', 'ltensor3', 'vector', 'matrix', 'tensor3',
    'scan', 'softmax', 'get_set_subtensor', 'reshape', 'switch',
    'shared', 'function', 'grad', 'ifelse', 'bscalar', 'eq', 'where',

    'stable_log_softmax_3', 'stable_softmax_3'
)


# Numpy
import numpy
from numpy import exp, tanh, zeros, dot, mean, concatenate, log, zeros_like, \
    arange, sqrt, tile, roll, where, ones_like, equal as eq
import numpy.random
import warnings

numpy.random.seed(0)
using_theano = True
floatX = 'float32'


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def scan(fun, n_steps, sequences=(), outputs_info=(), non_sequences=()):
    outs = [[None] * n_steps for _ in range(len(outputs_info))]
    upds = None
    sequences = list(sequences)
    non_sequences = list(non_sequences)
    outputs_info = list(outputs_info)
    for i in range(n_steps):
        seq = [apara[i] for apara in sequences]
        para = seq + outputs_info + non_sequences
        outputs = fun(*para)
        if isinstance(outputs, tuple):
            for j, out in enumerate(outputs):
                outs[j][i] = out
                outputs_info[j] = out
        else:
            outs[0][i] = outputs
            outputs_info[0] = outputs
    if isinstance(outputs, tuple):
        return outs, upds
    return outs[0], upds


def softmax(x, axis=1):
    e_xm = exp(x - x.max(axis=axis, keepdims=True))
    return e_xm / e_xm.sum(axis=axis, keepdims=True)


def get_set_subtensor(t, idx, v):
    t[idx] = v
    return t


def reshape(x, *shape):
    return x.reshape(*shape)


def switch(cond, if_true, if_false):
    warnings.warn('WARNING! buggy `switch` implementation.')
    if cond:
        return if_true
    else:
        return if_false


def ifelse(cond, if_true, if_false):
    if cond:
        return if_true
    else:
        return if_false


class SharedVariable(numpy.ndarray):
    def __new__(cls, value, name=None):
        obj = numpy.ndarray.__new__(cls, value.shape, value.dtype, None, 0, None, None)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)

    def get_value(self):
        return self.copy()

    def set_value(self, value):
        value = numpy.array(value)
        self.resize(value.shape, refcheck=False)
        numpy.copyto(self, value)
        return self


def shared(value, name=None, strict=False, allow_downcast=None, **kwargs):
    return SharedVariable(value, name)


def function(inputs, outputs, updates=None, givens=None):
    raise NotImplementedError


def bscalar(name):
    raise NotImplementedError

def lvector(name):
    raise NotImplementedError


def lmatrix(name):
    #raise NotImplementedError
    pass


def ltensor3(name):
    raise NotImplementedError


def vector(name):
    raise NotImplementedError


def matrix(name):
    #raise NotImplementedError
    pass


def tensor3(name):
    raise NotImplementedError


def grad(cost, wrt):
    raise NotImplementedError


# Theano
if using_theano:
    import theano
    import theano.tensor
    from theano.ifelse import ifelse

    floatX = theano.config.floatX

    scan = theano.scan
    shared = theano.shared
    function = theano.function

    grad = theano.gradient.grad

    bscalar = theano.tensor.bscalar
    lvector = theano.tensor.lvector
    lmatrix = theano.tensor.lmatrix
    ltensor3 = theano.tensor.ltensor3
    vector = theano.tensor.vector
    matrix = theano.tensor.matrix
    tensor3 = theano.tensor.tensor3

    tanh = theano.tensor.tanh
    zeros = theano.tensor.zeros
    concatenate = theano.tensor.concatenate
    mean = theano.tensor.mean
    dot = theano.tensor.dot
    exp = theano.tensor.exp
    log = theano.tensor.log
    zeros_like = theano.tensor.zeros_like
    ones_like = theano.tensor.ones_like
    arange = theano.tensor.arange
    switch = theano.tensor.switch
    sqrt = theano.tensor.sqrt
    tile = theano.tensor.tile
    roll = theano.tensor.roll
    where = theano.tensor.where
    eq = theano.tensor.eq
    sigmoid = theano.tensor.nnet.sigmoid
    softmax = theano.tensor.nnet.softmax

    def reshape(x, *shape):
        return x.reshape(shape)

    def get_set_subtensor(t, idx, v):
        return theano.tensor.set_subtensor(t[idx], v)


def stable_log_softmax_3(x):
    """Calculate log softmax in numerically stable way
    """
    x_max = x.max(axis=2, keepdims=True)
    x_safe = x - x_max
    x_safe_exp = exp(x_safe)
    x_safe_normalizer = x_safe_exp.sum(axis=2, keepdims=True)
    log_x_safe_normalizer = log(x_safe_normalizer)
    log_x_softmax = x_safe - log_x_safe_normalizer
    return log_x_softmax  # + log(x.shape[2]) # Bigarren zatia behar al da?


def stable_softmax_3(x):
    """Calculate softmax in numerically stable way
    """
    return reshape(softmax(reshape(x, x.shape[0] * x.shape[1], x.shape[2])), *x.shape)
