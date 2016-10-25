
from __future__ import print_function, division, absolute_import, unicode_literals
import logging
import copy

from .parameters import Parametric
from . import zeano
from .zeano import floatX, function, lmatrix, lvector, matrix, tensor3

import numpy as np

info = logging.info


class Sampler(Parametric):
    def __init__(self, model, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.hyperparameter('max_length', 60, "Maximum length of the generated translation.")
        self.model = model
        info('creating sampler...')
        self._sampler = self.get_sampler()
        info('created sampler')

    def get_sampler(self):
        if zeano.using_theano:
            x = lmatrix('x')
            x_mask = matrix('x_mask')
            word_i = lvector('word_i', 'int64')
            ctx_s = tensor3('ctx')
            state = matrix('state')

            ctx = self.model.encoder.encode(x, x_mask)
            init = self.model.decoder.decoder.init(ctx, x_mask)
            next_best = self.model.decoder.next_best(x_mask, word_i, ctx_s, state)

            self.init_fn = function((x, x_mask), (ctx, init))
            self.next_fn = function((x_mask, word_i, ctx_s, state), next_best)
        else:
            def init_fn(x, x_mask):
                ctx = self.model.encoder.encode(x, x_mask)
                return ctx, self.model.decoder.decoder.init(ctx, x_mask)

            self.init_fn = init_fn
            self.next_fn = self.model.decoder.next_best

    def sample(self, sentences, raw=False):
        if isinstance(sentences, (list, tuple)):
            return [self.sample(sentence, raw=raw) for sentence in sentences]

        if raw:
            sentences = u" ".join(map(str, self.model.preprocess(sentences)))
        ids = self.model.prepare_sample(sentences)
        x, mask = self.model.prepare_batch([ids])
        sample, score = self.do_sample(x, mask)
        # Remove the </s> symbol
        return self.model.tokenizer.join(self.model.decoder.target_vocabulary_r[w] for w in sample[:-1])

    def do_sample(self, x, mask):
        raise NotImplementedError


class StochasticSampler(Sampler):
    def do_sample(self, x, mask):
        sample = []
        sample_score = 0

        ctx, next_state = self.init_fn(x, mask)
        max_length = self.max_length if self.max_length else x.shape[0] * 2
        next_w = np.array([self.model.decoder.target_vocabulary['<s>']], dtype='int64')

        for ii in range(max_length):
            next_w, next_p, next_state = self.next_fn(mask, next_w, ctx, next_state)
            sample.append(next_w)
            sample_score += next_p
            if next_w == 0:
                break
        return sample, sample_score


class BeamSearchSampler(Sampler, Parametric):
    def __init__(self, model, **kwargs):
        Parametric.__init__(self, **kwargs)
        self.hyperparameter('beam_size', 6, "Beam size for the beam search.")
        self.hyperparameter('with_align', False, "Return also alignment.")
        Sampler.__init__(self, model, **kwargs)

    def get_sampler(self):
        if zeano.using_theano:
            x = lmatrix('x')
            x_mask = matrix('x_mask')
            word_i = lvector('word_i')
            ctx_s = tensor3('ctx')
            state = tensor3('state')

            ctx = self.model.encoder.encode(x, x_mask)
            init = self.model.decoder.decoder.init(ctx, x_mask)
            if self.with_align:
                probabilities = self.model.decoder.next_probabilities(x_mask, word_i, ctx_s, state, self.with_align)
            else:
                probabilities = self.model.decoder.next_probabilities(x_mask, word_i, ctx_s, state, self.with_align)
            self.init_fn = function((x, x_mask), (ctx, init))
            self.next_fn = function((x_mask, word_i, ctx_s, state), probabilities)
        else:
            def init_fn(x, x_mask):
                ctx = self.model.encoder.encode(x, x_mask)
                return ctx, self.model.decoder.init(ctx, x_mask)

            self.init_fn = init_fn
            self.next_fn = self.model.decoder.next_probabilities

    def do_sample(self, x, mask):
        sample = []
        sample_score = []
        alignments = []

        live_k = 1
        dead_k = 0

        if self.with_align:
            hyp_alignments = [[]]
        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k, dtype=floatX)

        ctx0, next_state = self.init_fn(x, mask)
        max_length = self.max_length if self.max_length else x.shape[0] * 2
        next_w = np.array([self.model.decoder.target_vocabulary['<s>']], dtype='int64')

        for ii in range(max_length):
            ctx = np.tile(ctx0, [live_k, 1])
            if self.with_align:
                next_p, next_state, ai = self.next_fn(mask.repeat(next_w.shape[0], axis=1), next_w, ctx, next_state)
                for line, align in zip(hyp_alignments, ai.T):
                    line.append(align)
            else:
                next_p, next_state = self.next_fn(mask.repeat(next_w.shape[0], axis=1), next_w, ctx, next_state)

            cand_scores = hyp_scores[:, None] - np.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(self.beam_size - dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat // voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = np.zeros(self.beam_size - dead_k).astype(floatX)
            new_hyp_states = []
            if self.with_align:
                new_hyp_alignments = []
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(next_state[:, ti:ti+1, :])
                if self.with_align:
                    new_hyp_alignments.append(hyp_alignments[ti])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            if self.with_align:
                hyp_alignments = []

            for idx in range(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                    if self.with_align:
                        alignments.append(new_hyp_alignments[idx])
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    if self.with_align:
                        hyp_alignments.append(new_hyp_alignments[idx][:])
            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= self.beam_size:
                break

            next_w = np.array([w[-1] for w in hyp_samples])
            next_state = np.concatenate(hyp_states, axis=1)

        # dump every remaining one
        if live_k > 0:
            for idx in range(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                if self.with_align:
                    alignments.append(hyp_alignments[idx][:])

        if self.with_align:
            return (max(zip(sample_score, sample, alignments)))[::-1]
        else:
            return (max(zip(sample_score, sample)))[::-1]

    def sample(self, sentences, raw=False):
        if isinstance(sentences, (list, tuple)):
            return [self.sample(sentence, raw=raw) for sentence in sentences]

        if raw:
            sentences = u" ".join(map(str, self.model.preprocess(sentences)))
        ids = self.model.prepare_sample(sentences)
        x, mask = self.model.prepare_batch([ids])
        if self.with_align:
            alignment, sample, score = self.do_sample(x, mask)
        else:
            sample, score = self.do_sample(x, mask)
        # Remove the </s> symbol
        sentence = self.model.tokenizer.join(self.model.decoder.target_vocabulary_r[w] for w in sample[:-1])
        if self.with_align:
            return sentence, alignment
        else:

            return sentence


samplers = {'stochastic-sampler': StochasticSampler, 'beam-search-sampler': BeamSearchSampler}
