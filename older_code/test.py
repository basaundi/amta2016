
from __future__ import print_function, division, absolute_import, unicode_literals
import os.path
import pickle
from io import open
from ann.samplers import BeamSearchSampler
from ann.data import Data
from multi_bleu import print_multi_bleu, tokenize_lower


def split_join(tokenizer, sentences):
    return [tokenizer.join(tokenizer.split(s.strip())) for s in sentences]


def main(arg_configuration, best=False, validation=True, output=None, head=None):
    path = os.path.dirname(arg_configuration)

    with open(arg_configuration, 'rb') as fd:
        configuration = pickle.load(fd)

    model = configuration['model_cls']()
    if best:
        model.load(os.path.join(path, 'model.pkl.best'))
    else:
        model.load(os.path.join(path, 'model.pkl'))
    sampler = BeamSearchSampler(model, beam_size=12)

    # FIXME: pre-configure validation set path
    if validation:
        test_set = Data(os.path.join(os.path.dirname(arg_configuration), '../valid.set'))
    else:
        test_set = Data(os.path.join(os.path.dirname(arg_configuration), '../test.set'))
    if head:
        test_set = test_set[:head]
    ss, st = map(list, zip(*iter(test_set)))
    st = list(split_join(model.tokenizer, st))

    translations = []
    for i, sentence in enumerate(ss):
        translations.append(sampler.sample(sentence))
        if (i % 10) == 0:
            print('translated {}/{} samples.'.format(i, len(ss)))

    if output:
        with open("{}.ref".format(output), "w", encoding="utf-8") as fdr:
            for r in st:
                print(unicode(r), file=fdr)
        with open(output, "w", encoding="utf-8") as fdt:
            for t in translations:
                print(unicode(t), file=fdt)

    print_multi_bleu(translations, (st,), tokenize_lower, 4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration')
    parser.add_argument('--output')
    parser.add_argument('--best', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--head', type=int)
    arguments = parser.parse_args()
    main(arguments.configuration, arguments.best, arguments.validation, arguments.output, arguments.head)
