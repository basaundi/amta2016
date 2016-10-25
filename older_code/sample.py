
from __future__ import print_function, division, absolute_import, unicode_literals
import os.path
import pickle
from ann.samplers import BeamSearchSampler
from io import open


def sample_lines(arg_configuration, lines):
    path = os.path.dirname(arg_configuration)

    with open(arg_configuration, 'rb') as fd:
        configuration = pickle.load(fd)

    model = configuration['model_cls']()
    model.load(os.path.join(path, 'model.pkl'))
    sampler = BeamSearchSampler(model)

    for line in lines:
        print(sampler.sample(line))


def main(arg_configuration, demo=False):
    lines = open(0, encoding="utf-8")  # sys.stdin
    if demo:
        lines = '''Oui .
        Merci beaucoup .
        Ordre des travaux'''.splitlines()
    return sample_lines(arg_configuration, lines)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('configuration')
    parser.add_argument('--demo', action='store_true')
    arguments = parser.parse_args()

    main(arguments.configuration, arguments.demo)
