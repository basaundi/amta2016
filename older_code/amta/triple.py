#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function, division, absolute_import, unicode_literals
from collections import OrderedDict, deque
from itertools import islice
import time
import numpy as np
from scipy.special import expit
import theano

from ann.optimizers import Adadelta
from ann.parameters import Parametric
from ann.trainer import Trainer
from ann.scheduler import BatchScheduler
from ann.zeano import function, grad, floatX, dot, tanh
from ann.layers import GRUCond, Random

from .bpe_models import ModelBPE
from .models import MonolingualWordModel, Encoder, Decoder, MonolingualSubWordModel


class UniformEncoder(Encoder):
    def encode(self, x, mask):
        x_emb = self.W_enc_emb[x] + self.b_enc_emb
        h = (self.f_encoder.apply(x_emb, mask)
             + self.b_encoder.apply(x_emb[::-1], mask[::-1])[::-1])
        return h


class ProjectionEncoder(Encoder):
    def __init__(self, **kwargs):
        super(ProjectionEncoder, self).__init__(**kwargs)
        self.parameter('W_an', Random(2*self.hidden_layer_size, 2*self.hidden_layer_size))
        self.parameter('b_an', Random(2*self.hidden_layer_size))

    def encode(self, x, mask):
        h = super(ProjectionEncoder, self).encode(x, mask)
        h_projected = tanh(dot(h, self.W_an) + self.b_an)
        return h_projected


class UniformDecoder(Decoder):
    def __init__(self, **kwargs):
        super(UniformDecoder, self).__init__(**kwargs)
        dim = self.hidden_layer_size
        dim_word = self.embedding_size
        self.parameter('decoder', GRUCond(dim_word=dim_word, dim=dim, ctx_dim=dim), override=True)
        self.parameter('Co', Random(dim, dim), override=True)


class AugmentedTranslationModel(Parametric):
    """
    Translation model trained on SRC -> TGT language pair,
    augmented on SRC -> SRC and AUX -> TGT pairs.
    """
    submodel_cls = MonolingualWordModel

    def __init__(self, **kwargs):
        super(AugmentedTranslationModel, self).__init__(**kwargs)
        self.parameter('model_src_tgt', self.submodel_cls(encoder_cls=ProjectionEncoder,
                                                          decoder_cls=Decoder, **kwargs))
        self.parameter('model_aux_tgt', self.submodel_cls(encoder_cls=ProjectionEncoder,
                                                          decoder_cls=Decoder, **kwargs))
        self.parameter('model_src_src', self.submodel_cls(encoder_cls=ProjectionEncoder,
                                                          decoder_cls=Decoder, **kwargs))
        self.model_aux_tgt.decoder = self.model_src_tgt.decoder
        self.model_src_src.encoder = self.model_src_tgt.encoder
        self.encoder = self.model_src_tgt.encoder
        self.decoder = self.model_src_tgt.decoder
        self.tokenizer = self.model_src_tgt.tokenizer
        self.shared_by = 1
        self.model_aux_tgt.decoder.shared_by = 2
        self.model_src_src.encoder.shared_by = 2

    def prepare_sample(self, *args):
        return self.model_src_tgt.prepare_sample(*args)

    def prepare_batch(self, *args):
        return self.model_src_tgt.prepare_batch(*args)

    def set_target_vocabulary(self, words):
        return self.model_src_tgt.set_target_vocabulary(words)

    def get_target_vocabulary(self):
        return self.model_src_tgt.get_target_vocabulary()

    def analyze_data(self, multi_data):
        src_tgt, aux_tgt, src_src = multi_data[:]
        self.model_src_tgt.analyze_data(src_tgt)
        self.model_aux_tgt.analyze_data(aux_tgt)
        self.model_src_src.analyze_data(src_src)

    def flat_parameters(self, p=None):
        if not p:
            parameters = super(AugmentedTranslationModel, self).flat_parameters()
            unique_p = {}
            seen = set()
            for k, v in parameters.items():
                if id(v) not in seen:
                    seen.add(id(v))
                    unique_p[k] = v
            return unique_p
        return super(AugmentedTranslationModel, self).flat_parameters(p)


class AugmentedSubWordTranslationModel(AugmentedTranslationModel):
    submodel_cls = MonolingualSubWordModel


class AugmentedBPESubWordTranslationModel(AugmentedTranslationModel):
    submodel_cls = ModelBPE


class SubBatchScheduler(BatchScheduler):
    def __init__(self, parent, model, prefix='train', **kwargs):
        super(SubBatchScheduler, self).__init__(model, prefix, **kwargs)
        self.parent = parent

    def _set_vocabulary(self):
        pass


class MultiScheduler(Parametric):
    def __init__(self, model, **kwargs):
        super(MultiScheduler, self).__init__(**kwargs)
        self.model = model
        self.parameter('src_tgt_scheduler', SubBatchScheduler(self, model.model_src_tgt, prefix='src'))
        self.parameter('aux_tgt_scheduler', SubBatchScheduler(self, model.model_aux_tgt, prefix='aux'))
        self.parameter('src_src_scheduler', BatchScheduler(model.model_src_src, prefix='mono'))

    def prepare_data(self, multi_data):
        data_src_tgt, data_aux_tgt, data_src_src = multi_data[0:3]
        self.src_tgt_scheduler.prepare_data(data_src_tgt)
        self.aux_tgt_scheduler.prepare_data(data_aux_tgt)
        self.src_src_scheduler.prepare_data(data_src_src)

    def get_next(self):
        return (self.src_tgt_scheduler.get_next(),
                self.aux_tgt_scheduler.get_next(),
                self.src_src_scheduler.get_next())

    def set_vocabulary(self):
        pass

class JointOptimizer(Adadelta):
    def __init__(self, model, **kwargs):
        super(JointOptimizer, self).__init__(model, **kwargs)
        self.hyperparameter('running_average_length', 30)
        self.hyperparameter('strictness', 50.0)

        self.coefficients_value = np.array([0.334, 0.333, 0.333])
        self.coefficients_mask = np.array([0.0, 1.0, 0.0])
        self.parameter('coefficients', np.array([0.334, 0.333, 0.333]))
        self.main_model = 0

        # save gradient for every parameter in the models (no repeat)
        parameters = self.model.flat_parameters()
        for name, param in parameters.items():
            if param.get_shared_by() > 1:
                self.parameter('{}_gradient'.format(name), np.zeros([0] * param.ndim))

        self.previous_costs = (deque([10, 10], self.running_average_length),
                               deque([10, 10], self.running_average_length),
                               deque([10, 10], self.running_average_length))

    def reset(self):
        super(JointOptimizer, self).reset()
        parameters = self.model.flat_parameters()
        for name, param in parameters.items():
            if param.get_shared_by() > 1:
                getattr(self, '{}_gradient'.format(name)).set_value(param.get_value() * 0.0)

    def _remember_updates(self, cost, parameters, i, encoder, decoder):
        updates = []
        immediate_params = OrderedDict()
        immediate_grads = []
        for name, parameter in parameters.items():
            if not encoder and ".encoder." in name:
                continue
            if not decoder and ".decoder." in name:
                continue
            try:
                gradient = grad(cost=cost, wrt=parameter)
            except theano.gradient.DisconnectedInputError:
                continue
            if parameter.get_shared_by() == 2:
                if i != 0:
                    coefficient = self.coefficients[i]
                else:
                    if ".decoder." in name:
                        coefficient = 1.0 - self.coefficients[1]
                    else:
                        coefficient = 1.0 - self.coefficients[2]
                prev_gradient = getattr(self, '{}_gradient'.format(name))
                updates.append((prev_gradient, prev_gradient + coefficient * gradient))
            else:
                immediate_params[name] = parameter
                immediate_grads.append(gradient)
        clipped_grads = self.clip(immediate_grads)
        immediate_updates = self.updates(immediate_params, clipped_grads)
        return immediate_updates + updates

    def get_cost_fn(self, i, encoder=True, decoder=True):
        models = self.model.model_src_tgt, self.model.model_aux_tgt, self.model.model_src_src
        parameters = OrderedDict(self.model.flat_parameters())

        print('compiling cost function {}...'.format(i + 1))
        model = models[i]
        cost = model.cost(*model.cost_arguments)
        updates = self._remember_updates(cost, parameters, i, encoder, decoder)
        fn = function(model.cost_arguments, cost, updates=updates)

        def cost_fn(*args):
            c = fn(*args)
            self.previous_costs[i].append(float(c))
            return c
        return cost_fn

    def _new_coefficients(self):
        #m1 = np.array([np.nanmean(list(islice(d, None, len(d) // 2))) for d in self.previous_costs])
        m2 = np.array([np.nanmean(list(islice(d, len(d) // 2, None))) for d in self.previous_costs])
        strictness = self.strictness
        #strictness *= 2 - abs((m2 - m1) * 100 / self.running_average_length).clip(0, 1)
        diff = m2[self.main_model] - m2
        print("strictness: {}; averages: {!r}".format(strictness, m2))
        raw_co = (0.5 - expit(strictness * diff)) * 1.00 + 0.5
        # raw_co[1] = np.clip(raw_co[1], 0.5, 1.0)
        raw_co[2] = np.clip(raw_co[2], 0.0, 0.5)
        return raw_co * self.coefficients_mask

    def get_update_fn(self):
        print('compiling update function...')
        parameters = OrderedDict(self.model.flat_parameters())
        mean_param = OrderedDict()
        mean_grads = []
        zero_upds = []
        for name, parameter in parameters.items():
            if parameter.get_shared_by() > 1:
                prev_gradient = getattr(self, '{}_gradient'.format(name))
                mean_param[name] = parameter
                mean_grads.append(prev_gradient)
                zero_upds.append((prev_gradient, prev_gradient * 0))
        grads = self.clip(mean_grads)
        updates = self.updates(mean_param, grads)
        fn = function(tuple(), tuple(), updates=updates + zero_upds)

        def update_fn():
            print("update fn()")
            fn()
            self.coefficients_value = self._new_coefficients()
            self.coefficients_value = np.dtype(floatX).type(self.coefficients_value)
            self.coefficients.set_value(self.coefficients_value)
            print("updated. New coefficients: {!r}".format(self.coefficients_value))
        return update_fn


class JointTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(JointTrainer, self).__init__(*args, **kwargs)
        self.perform_update = None
        self.cost_src_tgt_fn = None
        self.cost_aux_tgt_fn = None
        self.cost_src_src_fn = None
        self.using_src = False
        self.using_aux = True
        self.using_mono = False
        self.using_mono2 = False

    @property
    def epoch_f(self):
        return self.samples_done / self.scheduler.src_tgt_scheduler.total_samples

    def _pull_minibatch(self):
        minibatch = self.scheduler.get_next()
        prepared_batches = (self.model.model_src_tgt.prepare_batch(*minibatch[0]),
                            self.model.model_aux_tgt.prepare_batch(*minibatch[1]),
                            self.model.model_src_src.prepare_batch(*minibatch[2]))
        return prepared_batches

    def _before_minibatch(self, minibatches):
        self.last_minibatch = minibatches[0]
        self.last_minibatch_src_tgt, self.last_minibatch_aux_tgt, self.last_minibatch_src_src = minibatches[:]
        self.time_pulling_data += time.time() - self._time_last_minibatch

    def _process_minibatch(self, minibatches):
        minibatch_src_tgt, minibatch_aux_tgt, minibatch_src_src = minibatches[:]
        self.last_cost_src_tgt = 10.0
        self.last_cost_aux_tgt = 10.0
        self.last_cost_src_src = 10.0
        if self.using_src:
            self.last_cost_src_tgt = self.cost_src_tgt_fn(*minibatch_src_tgt)
        if self.using_aux:
            self.last_cost_aux_tgt = self.cost_aux_tgt_fn(*minibatch_aux_tgt)
        if self.using_mono:
            self.last_cost_src_src = self.cost_src_src_fn(*minibatch_src_src)
        self.perform_update()

    def _after_minibatch(self, minibatches):
        minibatch = minibatches[0]
        samples = minibatch[-1].shape[1]
        self.x_words_done += int(minibatch[1].sum()) - samples  # Discount EOS
        self.y_words_done += int(minibatch[3].sum()) - samples  # Discount EOS
        self.samples_done += samples
        self.time_training += time.time() - self._time_last_minibatch
        for observer in self.observers:
            observer.trigger()
        self._update_state()

    def _update_state(self):
        days = 24 * 60 * 60.0
        if self.time_training > 6.5 * days:
            if not self.using_mono2:
                self.cost_src_src_fn = self.optimizer.get_cost_fn(2, True, False)
                self.using_mono = True
                self.using_mono2 = True
                self.optimizer.coefficients_mask[2] = 1.0
        if self.time_training > 6.0 * days:
            if not self.using_mono and not self.using_mono2:
                self.cost_src_src_fn = self.optimizer.get_cost_fn(2, False, True)
                self.using_mono = True
        if self.time_training >= 0 * days:
            if not self.using_src:
                self.cost_src_tgt_fn = self.optimizer.get_cost_fn(0)
                self.using_src = True
                self.optimizer.coefficients_mask[1] = 1.0
        if self.time_training >= 0 * days:
            if not self.using_aux:
                self.cost_aux_tgt_fn = self.optimizer.get_cost_fn(1, True, True)
                self.using_aux = True

    def _get_cost_with_update(self):
        if self.perform_update:
            return
        print('compiling cost functions...')
        #self.cost_src_tgt_fn = self.optimizer.get_cost_fn(0)
        self.cost_aux_tgt_fn = self.optimizer.get_cost_fn(1)
        #self.cost_src_src_fn = self.optimizer.get_cost_fn(2)
        self.perform_update = self.optimizer.get_update_fn()

    def print_status(self):
        print("iteration #{} ({} epochs of SRC)".format(self.iterations_done + 1, self.epoch_f))
        if self.using_src:
            print("  [SRC] ({}) --> {}".format(self.last_minibatch_src_tgt[0].shape, self.last_cost_src_tgt))
        if self.using_aux:
            print("  [AUX] ({}) --> {}".format(self.last_minibatch_aux_tgt[0].shape, self.last_cost_aux_tgt))
        if self.using_mono:
            print("  [MON] ({}) --> {}".format(self.last_minibatch_src_src[0].shape, self.last_cost_src_src))
        print("({:.2f}%) time training: {:.2f}; pulling: {:.2f}".format(
            100 * self.time_pulling_data / self.time_training, self.time_training, self.time_pulling_data
        ))

    def is_done(self):
        # FIXME: txkundu dena
        return self.using_mono2
        return self.iterations_done >= self.target_iterations
