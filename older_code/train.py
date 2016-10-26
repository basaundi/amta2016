
from __future__ import print_function, division, absolute_import, unicode_literals
import os
import os.path
import pickle
from ann.observers import Saver, Validator, ProgressLogger, Outputter
from ann.data import Data
from ann.samplers import BeamSearchSampler


def hooks(trainer_cls):
    # FIXME: pre-configure validation set path
    valid_set = Data('../valid.set')

    if hasattr(trainer_cls, "print_status"):
        fields = ('iterations_done', 'epochs_done', 'samples_done',
                            'x_words_done', 'y_words_done', 'time_pulling_data', 'time_training',
                            'last_cost_src_tgt', 'last_cost_aux_tgt', 'last_cost_src_src')
    else:
        fields = ('iterations_done', 'epochs_done', 'samples_done',
                            'x_words_done', 'y_words_done', 'time_pulling_data', 'time_training', 'last_cost')
    return (
        Saver(every_x_training_seconds=30 * 60),
        Validator('validation.csv', valid_set, BeamSearchSampler, every_n_iterations=1500),
        ProgressLogger('training.csv', fields, every_n_iterations=1),
        Outputter(every_n_iterations=1)
    )


def main(arg_configuration, prepare=False):
    path = os.path.dirname(arg_configuration)
    os.chdir(path)

    with open('configuration.pkl', 'rb') as fd:
        configuration = pickle.load(fd)

    model = configuration['model_cls']()
    data = configuration['data_cls']()
    trainer_cls = configuration['trainer_cls']
    trainer = trainer_cls(model, data, configuration['optimizer_cls'],
                          configuration['scheduler_cls'], hooks(trainer_cls))
    trainer.load('trainer.pkl')
    if prepare:
        trainer.prepare_data()
    print('Start training...')
    trainer.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration')
    parser.add_argument('--prepare', action='store_true')
    arguments = parser.parse_args()
    main(arguments.configuration, arguments.prepare)
