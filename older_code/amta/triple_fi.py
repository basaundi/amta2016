
from collections import OrderedDict
from ann.parameters import Parametric
from ann.scheduler import BatchScheduler
from ann.zeano import grad
import theano
from .triple import (AugmentedTranslationModel, UniformEncoder, UniformDecoder, JointOptimizer, MonolingualSubWordModel,
                     MultiScheduler, JointTrainer, SubBatchScheduler)
from .bpe_models import ModelBPE


class AugmentedTranslationModel2(AugmentedTranslationModel):
    def __init__(self, **kwargs):
        Parametric.__init__(self, **kwargs)
        self.parameter('model_src_tgt', self.submodel_cls(encoder_cls=UniformEncoder,
                                                          decoder_cls=UniformDecoder, **kwargs))
        self.parameter('model_src_aux', self.submodel_cls(encoder_cls=UniformEncoder,
                                                          decoder_cls=UniformDecoder, **kwargs))
        self.parameter('model_tgt_tgt', self.submodel_cls(encoder_cls=UniformEncoder,
                                                
        self.model_aux_tgt = self.model_src_aux
        self.model_src_src = self.model_tgt_tgt
        self.model_src_aux.encoder = self.model_src_tgt.encoder
        self.model_tgt_tgt.decoder = self.model_src_tgt.decoder
        self.encoder = self.model_src_tgt.encoder
        self.decoder = self.model_src_tgt.decoder
        self.tokenizer = self.model_src_tgt.tokenizer
        self.shared_by = 1
        self.decoder.shared_by = 2
        self.encoder.shared_by = 2


class AugmentedSubWordTranslationModel2(AugmentedTranslationModel2):
    submodel_cls = MonolingualSubWordModel


class AugmentedBPESubWordTranslationModel2(AugmentedTranslationModel2):
    submodel_cls = ModelBPE


class JointOptimizer2(JointOptimizer):
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
                    if ".encoder." in name:
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


class JointTrainer2(JointTrainer):
    def is_done(self):
        days = 24 * 60 * 60.0
        return self.time_training > 6.2 * days
        return self.iterations_done >= self.target_iterations

    def _update_state(self):
        days = 24 * 60 * 60.0
        if self.time_training > 6.1 * days:
            if not self.using_mono2:
                self.cost_src_src_fn = self.optimizer.get_cost_fn(2, False, True)
                self.using_mono = True
                self.using_mono2 = True
                self.optimizer.coefficients_mask[2] = 1.0
        if self.time_training > 6.0 * days:
            if not self.using_mono and not self.using_mono2:
                self.cost_src_src_fn = self.optimizer.get_cost_fn(2, True, False)
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


class MultiScheduler2(MultiScheduler):
    def __init__(self, model, **kwargs):
        super(MultiScheduler2, self).__init__(model, **kwargs)
        self.parameter('src_tgt_scheduler', SubBatchScheduler(self, model.model_src_tgt, prefix='src'), override=True)
        self.parameter('aux_tgt_scheduler', BatchScheduler(model.model_aux_tgt, prefix='aux'), override=True)
        self.parameter('src_src_scheduler', SubBatchScheduler(self, model.model_src_src, prefix='mono'), override=True)

    def set_vocabulary(self):
        pass
