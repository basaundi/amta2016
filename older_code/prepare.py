#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function, division, absolute_import, unicode_literals
import pickle
import os.path as osp
import os
from ann.data import MultiData, Data
from amta.amta import SubwordModel as ModelSW, ModelBPE
from amta.amta import AugmentedSubWordTranslationModel

from ann.optimizers import Adadelta as OptimizerCls
from ann.trainer import Trainer
from ann.scheduler import BatchScheduler as SchedulerCls

from amta.triple import (JointTrainer,
                                MultiScheduler,
                                JointOptimizer)


def prepare(case_name, trainer):
    print("Preparing training folder `{}`...".format(case_name))
    try:
        os.mkdir(case_name)
    except OSError:
        print("... folder already exists. Skipping.")
        return

    trainer_cls = type(trainer)
    model_cls = type(trainer.model)
    data_cls = type(trainer.data)

    configuration = dict(
        name=case_name,
        trainer_cls=trainer_cls,
        model_cls=model_cls,
        optimizer_cls=trainer.optimizer_cls,
        scheduler_cls=trainer.scheduler_cls,
        data_cls=data_cls
    )

    with open(osp.join(case_name, 'configuration.pkl'), 'wb') as fd:
        pickle.dump(configuration, fd)
    trainer.save(osp.join(case_name, 'trainer.pkl'))


def prepare_experiments():
    hooks = tuple()
    data_src_tgt = Data('train.set')
    data_aux_tgt = Data('train.aux.set')
    data_src_src = Data('train.mono.set')

    mini_data = data_src_tgt[:15000]
    data_210 = data_src_tgt[:210000]
    data_420 = data_src_tgt[:420000]

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = ModelSW(**hyperparameters)
    trainer = Trainer(model, mini_data, OptimizerCls, SchedulerCls, hooks, **hyperparameters)
    prepare('mini', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = ModelBPE(**hyperparameters)
    trainer = Trainer(model, data_210, OptimizerCls, SchedulerCls, hooks, **hyperparameters)
    #prepare('bpe210', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = ModelBPE(**hyperparameters)
    trainer = Trainer(model, data_420, OptimizerCls, SchedulerCls, hooks, **hyperparameters)
    #prepare('bpe420', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = ModelSW(**hyperparameters)
    trainer = Trainer(model, data_210, OptimizerCls, SchedulerCls, hooks, **hyperparameters)
    #prepare('single210', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = ModelSW(**hyperparameters)
    trainer = Trainer(model, data_420, OptimizerCls, SchedulerCls, hooks, **hyperparameters)
    #prepare('single420', trainer)

    mdata15 = MultiData(data_src_tgt[:15000], data_aux_tgt[:1000000], data_src_src[:1000000])
    mdata210 = MultiData(data_src_tgt[:210000], data_aux_tgt[:4000000], data_src_src[:4000000])
    mdata420 = MultiData(data_src_tgt[:420000], data_aux_tgt[:4000000], data_src_src[:4000000])

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = AugmentedSubWordTranslationModel(**hyperparameters)
    trainer = JointTrainer(model, mdata15, JointOptimizer, MultiScheduler, hooks, **hyperparameters)
    prepare('subword_defr-en_15', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = AugmentedSubWordTranslationModel(**hyperparameters)
    trainer = JointTrainer(model, mdata210, JointOptimizer, MultiScheduler, hooks, **hyperparameters)
    #prepare('subword_defr-en_210', trainer)

    hyperparameters = {'maximum_vocabulary_size': 30000}
    model = AugmentedSubWordTranslationModel(**hyperparameters)
    trainer = JointTrainer(model, mdata420, JointOptimizer, MultiScheduler, hooks, **hyperparameters)
    #prepare('subword_defr-en_420', trainer)


if __name__ == '__main__':
    name = 'amta16'
    import env
    cwd = env.set_workspace(name=name, persistent=True, scriptname=False)
    print(cwd)
    prepare_experiments()
