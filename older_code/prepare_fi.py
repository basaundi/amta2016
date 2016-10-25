#!/usr/bin/python
# -*- coding: UTF-8 -*-


from __future__ import print_function, division, absolute_import, unicode_literals
from ann.data import MultiData, Data
from ann.optimizers import Adadelta as OptimizerCls
from ann.trainer import Trainer
from ann.scheduler import BatchScheduler as SchedulerCls

from amta.amta import SubwordModel as ModelSW, ModelBPE
from amta.amta import AugmentedSubWordTranslationModel2
from amta.triple_fi import JointTrainer2, MultiScheduler2, JointOptimizer2
from prepare import prepare


def prepare_experiments():
    hooks = tuple()
    data_src_tgt = Data('train.set')
    data_aux_tgt = Data('train.aux.set')
    data_src_src = Data('train.mono.set')

    mini_data = data_src_tgt[:15000]
    data_100 = data_src_tgt[:100000]
    data_200 = data_src_tgt[:200000]

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = ModelSW(**hyperparameters)
    trainer = Trainer(model, mini_data, OptimizerCls, SchedulerCls, hooks, **hyperparameters)
    prepare('mini', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = ModelBPE(**hyperparameters)
    trainer = Trainer(model, data_100, OptimizerCls, SchedulerCls, hooks, **hyperparameters)
    prepare('bpe100', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = ModelBPE(**hyperparameters)
    trainer = Trainer(model, data_200, OptimizerCls, SchedulerCls, hooks, **hyperparameters)
    prepare('bpe200', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = ModelSW(**hyperparameters)
    trainer = Trainer(model, data_100, OptimizerCls, SchedulerCls, hooks, **hyperparameters)
    prepare('single100', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = ModelSW(**hyperparameters)
    trainer = Trainer(model, data_200, OptimizerCls, SchedulerCls, hooks, **hyperparameters)
    prepare('single200', trainer)

    mdata15 = MultiData(data_src_tgt[:15000], data_aux_tgt[:1000000], data_src_src[:1000000])
    data_100 = MultiData(data_src_tgt[:100000], data_aux_tgt[:4000000], data_src_src[:4000000])
    mdata200 = MultiData(data_src_tgt[:200000], data_aux_tgt[:4000000], data_src_src[:4000000])

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = AugmentedSubWordTranslationModel2(**hyperparameters)
    trainer = JointTrainer2(model, mdata15, JointOptimizer2, MultiScheduler2, hooks, **hyperparameters)
    prepare('subword_en-fifr_15', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = AugmentedSubWordTranslationModel2(**hyperparameters)
    trainer = JointTrainer2(model, data_100, JointOptimizer2, MultiScheduler2, hooks, **hyperparameters)
    prepare('subword_en-fifr_100', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = AugmentedSubWordTranslationModel2(**hyperparameters)
    trainer = JointTrainer2(model, mdata200, JointOptimizer2, MultiScheduler2, hooks, **hyperparameters)
    prepare('subword_en-fifr_200', trainer)


if __name__ == '__main__':
    name = 'amta16_fi'
    import env
    cwd = env.set_workspace(name=name, persistent=True, scriptname=False)
    print(cwd)
    prepare_experiments()
