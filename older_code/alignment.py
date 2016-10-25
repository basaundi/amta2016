
from __future__ import print_function, division, absolute_import, unicode_literals
import os.path
import pickle
from io import open
from ann.samplers import BeamSearchSampler

import numpy as np
import matplotlib.pyplot as plt


def plot(alignment, source, translation):
    fig, ax = plt.subplots()
    plt.style.use('classic')
    ax.set_axis_bgcolor('red')
    image = np.matrix(alignment)
    ax.imshow(image, cmap="hot", interpolation='none')
    ax.xaxis.set_ticks_position('top')
    source_tok = source.split() + ["</s>"]
    translation_tok = translation.split() + ["</s>"]
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['top'].set_position(('outward', 10))
    plt.xticks(range(image.shape[1]), source_tok, rotation=90)
    plt.yticks(range(image.shape[0]), translation_tok)


def main(arg_configuration, source, best=False):
    path = os.path.dirname(arg_configuration)

    with open(arg_configuration, 'rb') as fd:
        configuration = pickle.load(fd)

    model = configuration['model_cls']()
    if best:
        model.load(os.path.join(path, 'model.pkl.best'))
    else:
        model.load(os.path.join(path, 'model.pkl'))

    sampler = BeamSearchSampler(model, beam_size=12, with_align=True)
    translation, alignment = sampler.sample(source)

    print("SRC: {}".format(source))
    print("CND: {}".format(translation))

    source_tok = " ".join(t.replace(u"\u241f", u"_").replace(u"\u241d", u"") for t in model.tokenizer.split(source))
    translation_tok = " ".join(t.replace(u"\u241f", u"_").replace(u"\u241d", u"") for t in model.tokenizer.split(translation))
    plot(alignment, source_tok, translation_tok)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('--best', action='store_true')
    arguments = parser.parse_args()
    main(arguments.configuration, arguments.source, arguments.best)
