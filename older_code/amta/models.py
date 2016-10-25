from __future__ import print_function, division, absolute_import, unicode_literals
from collections import Counter
import numpy as np
from ann.parameters import Parametric
from ann.zeano import tanh, concatenate, dot, arange, reshape, \
    stable_log_softmax_3, floatX, roll, get_set_subtensor, lmatrix, matrix, \
    softmax, shared, ifelse
from ann.layers import Random, GRU, GRUCond
from .tokenizers import SplitTokenizer, VowelTokenizer


class Encoder(Parametric):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.hyperparameter('embedding_size', 620)
        self.hyperparameter('hidden_layer_size', 1000, "Size of the hidden layer.")

        self.hyperparameter('source_vocabulary', {'</s>': 0, 'UNK': 1, '<s>': 2})
        self.hyperparameter('source_vocabulary_r', ['</s>', 'UNK', '<s>'])
        self.parameter('W_enc_emb', Random(len(self.source_vocabulary), self.embedding_size))
        self.parameter('b_enc_emb', Random(self.embedding_size))
        self.parameter('f_encoder', GRU(dim_word=self.embedding_size, dim=self.hidden_layer_size))
        self.parameter('b_encoder', GRU(dim_word=self.embedding_size, dim=self.hidden_layer_size))

    def encode(self, x, mask):
        x_emb = self.W_enc_emb[x] + self.b_enc_emb
        h = concatenate((
            self.f_encoder.apply(x_emb, mask),
            self.b_encoder.apply(x_emb[::-1], mask[::-1])[::-1]
        ), axis=2)
        return h


class Decoder(Parametric):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.hyperparameter('embedding_size', 620)
        self.hyperparameter('hidden_layer_size', 1000, "Size of the hidden layer.")

        self.hyperparameter('target_vocabulary', {'</s>': 0, 'UNK': 1, '<s>': 2})
        self.hyperparameter('target_vocabulary_r', ['</s>', 'UNK', '<s>'])

        dim = self.hidden_layer_size
        dim_word = self.embedding_size
        self.parameter('W_dec_emb', Random(len(self.target_vocabulary), dim_word))
        self.parameter('b_dec_emb', Random(dim_word))
        self.parameter('decoder', GRUCond(dim_word=dim_word, dim=dim, ctx_dim=2*dim))

        self.parameter('Uo', Random(dim, dim))
        self.parameter('Uob', Random(dim))
        self.parameter('Co', Random(2 * dim, dim))
        self.parameter('Vo', Random(dim_word, dim))

        self.parameter('Wo1', Random(dim / 2, dim_word))
        self.parameter('Wo2', Random(dim_word, len(self.target_vocabulary)))
        self.parameter('bWo', Random(len(self.target_vocabulary)))
        self.candidates = shared(np.array([], 'int64'), 'candidates')
        self.wid = shared(Random(len(self.target_vocabulary)).astype('int64'), 'wid')

    def decode(self, x_mask, y, y_mask, ctx):
        rolled_y = roll(y, 1, axis=0)
        rolled_y = get_set_subtensor(rolled_y, (0, slice(None)), self.target_vocabulary['<s>'])
        rolled_embs = self.W_dec_emb[rolled_y] + self.b_dec_emb

        init_state = self.decoder.init(ctx, x_mask)
        s, ci = self.decoder.apply(ctx, x_mask, rolled_embs, y_mask, init_state)

        # calculate probabilities
        rolled_states = roll(s, 1, axis=0)
        rolled_states = get_set_subtensor(rolled_states, (0, slice(None), slice(None)), init_state[0])
        score = self._scores(ci, rolled_states, rolled_embs)
        logsoft = stable_log_softmax_3(score)
        s_cost = self._mangle_y_index(logsoft, y)
        weighted_y_mask = y_mask / y_mask.sum(0)
        cost = -(s_cost * weighted_y_mask).sum(0)
        return cost.mean()

    def _mangle_y_index(self, values, y):
        y_flat = y.flatten()
        # FIXME: Nnapa
        if False:
            idx = arange(y_flat.shape[0]) * values.shape[-1] + self.wid[y_flat]
            return values.flatten()[idx].reshape(y.shape)
        else:
            idx = arange(y_flat.shape[0]) * values.shape[-1] + y_flat
            return values.flatten()[idx].reshape(y.shape)

    def _scores(self, ci, state, prev_word_emb):
        tni = self._target_vector(ci, state, prev_word_emb)
        ti = reshape(tni, tni.shape[0], tni.shape[1], tni.shape[2] // 2, 2).max(axis=-1)
        # FIXME: Nnapa
        if False:
            Wo2 = ifelse(self.candidates.shape[0], self.Wo2[:, self.candidates], self.Wo2)
            bWo = ifelse(self.candidates.shape[0], self.bWo[self.candidates], self.bWo)
        else:
            Wo2 = self.Wo2
            bWo = self.bWo
        scores = dot(dot(ti, self.Wo1), Wo2) + bWo
        return scores

    def _target_vector(self, ci, state, prev_word_emb):
        tni_a = dot(state, self.Uo) + self.Uob + dot(ci, self.Co)
        tni_b = dot(prev_word_emb, self.Vo)
        tni = tanh(tni_a + tni_b)
        return tni

    def next_probabilities(self, x_mask, prev_word, ctx, state, with_align=False):
        prev_word_emb = self.W_dec_emb[prev_word] + self.b_dec_emb
        upd_embs_i, rst_embs_i, inp_embs_i, cti_embs_i = self.decoder.gate_projections(prev_word_emb)

        # TODO: Move this out of the function. Maybe decoder.init?
        eb = dot(ctx, self.decoder.Ua)  # Projection of ctx
        if with_align:
            ci, ai = self.decoder.context_vector(ctx, x_mask, state[0], eb, cti_embs_i, True)
        else:
            ci = self.decoder.context_vector(ctx, x_mask, state[0], eb, cti_embs_i)

        score = self._scores(ci, state, prev_word_emb)
        probs = softmax(reshape(score, score.shape[1], score.shape[2]))

        rst, upd, inp = self.decoder.gate_embeddings(upd_embs_i, rst_embs_i, inp_embs_i, ci)
        new_state = self.decoder.update_state(x_mask[0], inp, rst, upd, state)
        if with_align:
            return probs, new_state, ai
        else:
            return probs, new_state

    def set_candidates(self, words):
        self.candidates.set_value(words)
        wid = (self.bWo.get_value() * 0.0).astype('int64')
        words_l = [w for w in words if w < len(wid)]
        wid[words_l] = range(len(words_l))
        self.wid.set_value(wid)


class MonolingualWordModel(Parametric):
    cost_arguments = (lmatrix('x'), matrix('x_mask'), lmatrix('y'), matrix('y_mask'))
    tokenizer_cls = SplitTokenizer

    def __init__(self, encoder_cls=Encoder, decoder_cls=Decoder, **kwargs):
        super(MonolingualWordModel, self).__init__(**kwargs)
        self.hyperparameter('embedding_size', 620)
        self.hyperparameter('hidden_layer_size', 1000, "Size of the hidden layer.")
        self.hyperparameter('maximum_vocabulary_size', float('inf'), "Maximum vocabulary size.")
        self.parameter('encoder', encoder_cls(**kwargs))
        self.parameter('decoder', decoder_cls(**kwargs))
        self.tokenizer = self.tokenizer_cls()

    def set_target_vocabulary(self, words):
        return False
        self.decoder.set_candidates(words)

    def get_target_vocabulary(self):
        return self.decoder.candidates.get_value()

    def analyze_data(self, data):
        counts_src = Counter()
        counts_tgt = Counter()
        for x, y in data:
            for w in self.tokenizer.split(x):
                counts_src[w] += 1
            for w in self.tokenizer.split(y):
                counts_tgt[w] += 1
        self.encoder.source_vocabulary_r = ['</s>', 'UNK', '<s>']
        self.decoder.target_vocabulary_r = ['</s>', 'UNK', '<s>']
        self.encoder.source_vocabulary_r.extend(k for k, v in counts_src.most_common())
        self.decoder.target_vocabulary_r.extend(k for k, v in counts_tgt.most_common())
        self.encoder.source_vocabulary = {k: i for i, k in enumerate(self.encoder.source_vocabulary_r)
                                          if i < self.maximum_vocabulary_size}
        self.decoder.target_vocabulary = {k: i for i, k in enumerate(self.decoder.target_vocabulary_r)
                                          if i < self.maximum_vocabulary_size}

        print('Vocabulary sizes (src, tgt):',
                len(self.encoder.source_vocabulary),
                len(self.decoder.target_vocabulary))

        self.encoder.W_enc_emb.set_value(Random(len(self.encoder.source_vocabulary), self.encoder.embedding_size))
        self.decoder.W_dec_emb.set_value(Random(len(self.decoder.target_vocabulary), self.encoder.embedding_size))
        self.decoder.Wo2.set_value(Random(self.decoder.embedding_size, len(self.decoder.target_vocabulary)))
        self.decoder.bWo.set_value(Random(len(self.decoder.target_vocabulary)))

    def prepare_sample(self, sample):
        UNK1 = self.encoder.source_vocabulary['UNK']
        UNK2 = self.decoder.target_vocabulary['UNK']

        if isinstance(sample, (list, tuple)) and len(sample) == 2:
            x, y = sample
            idx = tuple(self.encoder.source_vocabulary.get(w, UNK1) for w in self.tokenizer.split(x))
            idy = tuple(self.decoder.target_vocabulary.get(w, UNK2) for w in self.tokenizer.split(y))
            return idx, idy
        return tuple(self.encoder.source_vocabulary.get(w, UNK1) for w in self.tokenizer.split(sample))

    @staticmethod
    def _prepare_mask_pair(ids):
        # NOTE: A column per sentence
        ns = list(map(len, ids))
        # The extra character is for the EOS
        x = np.zeros((max(ns) + 1, len(ids)), dtype='int64')
        m = np.zeros_like(x).astype(floatX)
        for i, length in enumerate(ns):
            x[0:length, i] = ids[i]
            m[0:length + 1, i] = 1
        return x, m

    @classmethod
    def prepare_batch(cls, xs, ys=None):
        if not ys:
            return cls._prepare_mask_pair(xs)
        return cls._prepare_mask_pair(xs) + cls._prepare_mask_pair(ys)

    def cost(self, x, x_mask, y, y_mask):
        ctx = self.encoder.encode(x, x_mask)
        return self.decoder.decode(x_mask, y, y_mask, ctx)

    def next_probabilities(self, x_mask, prev_word, ctx, state, with_align=False):
        return self.decoder.next_probabilities(x_mask, prev_word, ctx, state, with_align)

    def next_best(self, mask, prev_word, ctx, state):
        probs, next_state = self.decoder.next_probabilities(mask, prev_word, ctx, state)
        new_words = probs.argmax(axis=-1)  # SoftmaxLayer
        new_probs = probs[arange(new_words.shape[0]), new_words.squeeze()]
        return new_words, new_probs, next_state


class MonolingualSubWordModel(MonolingualWordModel):
    tokenizer_cls = VowelTokenizer

    def _adapt_to_vocabulary(self, vocabulary, tokens):
        UNK = vocabulary['UNK']
        ids = []
        for token in tokens:
            if token in vocabulary:
                ids.append(vocabulary[token])
            else:
                unknown = True
                for trimming in self.tokenizer.trimmings(token):
                    if trimming in vocabulary:
                        ids.append(vocabulary[trimming])
                        unknown = False
                        break
                if unknown:
                    ids.append(UNK)
        return ids

    def prepare_sample(self, sample):
        if isinstance(sample, (list, tuple)) and len(sample) == 2:
            x, y = sample
            idx = self._adapt_to_vocabulary(self.encoder.source_vocabulary, self.tokenizer.split(x))
            idy = self._adapt_to_vocabulary(self.decoder.target_vocabulary, self.tokenizer.split(y))
            return idx, idy
        return self._adapt_to_vocabulary(self.encoder.source_vocabulary, self.tokenizer.split(sample))


class BilingualWordModel(MonolingualWordModel):
    cost_arguments = (lmatrix('x'), matrix('x_mask'), lmatrix('y'), matrix('y_mask'))

    def __init__(self, **kwargs):
        super(BilingualWordModel, self).__init__(**kwargs)
        self.hyperparameter('auxiliary_vocabulary', {'</s>': 0, 'UNK': 1, '<s>': 2})
        self.hyperparameter('auxiliary_vocabulary_r', ['</s>', 'UNK', '<s>'])
        self.parameter('W_enc_emb_aux', Random(len(self.auxiliary_vocabulary), self.embedding_size))
        self.parameter('b_enc_emb_aux', Random(self.embedding_size))
        self.parameter('f_encoder_aux', GRU(dim_word=self.embedding_size, dim=self.hidden_layer_size))
        self.parameter('b_encoder_aux', GRU(dim_word=self.embedding_size, dim=self.hidden_layer_size))

    def _encode_aux(self, x, mask):
        x_emb = self.W_enc_emb_aux[x] + self.b_enc_emb_aux
        h = concatenate((
            self.f_encoder_aux.apply(x_emb, mask),
            self.b_encoder_aux.apply(x_emb[::-1], mask[::-1])[::-1]
        ), axis=2)
        return h

    def cost_auxiliary(self, x, x_mask, y, y_mask):
        rolled_y = roll(y, 1, axis=0)
        rolled_y = get_set_subtensor(rolled_y, (0, slice(None)), self.target_vocabulary['<s>'])
        rolled_embs = self.W_dec_emb[rolled_y] + self.b_dec_emb
        ctx = self._encode_aux(x, x_mask)
        init_state = self.decoder.init(ctx, x_mask)
        s, ci = self.decoder.apply(ctx, x_mask, rolled_embs, y_mask, init_state)

        # calculate probabilities
        rolled_states = roll(s, 1, axis=0)
        rolled_states = get_set_subtensor(rolled_states, (0, slice(None), slice(None)), init_state[0])
        score = self._scores(ci, rolled_states, rolled_embs)
        logsoft = stable_log_softmax_3(score)
        s_cost = self._mangle_y_index(logsoft, y)
        weighted_y_mask = y_mask / y_mask.sum(0)
        cost = -(s_cost * weighted_y_mask).sum(0)
        return cost.mean()

    @classmethod
    def prepare_batch(cls, xs, ys=None, origin=0):
        if not ys:
            return cls._prepare_mask_pair(xs)
        return (origin,) + cls._prepare_mask_pair(xs) + cls._prepare_mask_pair(ys)

    def prepare_sample(self, sample, origin=None):
        if origin is None:
            return super(BilingualWordModel, self).prepare_sample(sample)

        UNK_S = self.source_vocabulary['UNK']
        UNK_A = self.auxiliary_vocabulary['UNK']
        UNK_T = self.target_vocabulary['UNK']
        if isinstance(sample, (list, tuple)) and len(sample) == 2:
            x, y = sample
            if origin == 0:
                idx = tuple(self.source_vocabulary.get(w, UNK_S) for w in self.tokenizer.split(x))
            else:
                idx = tuple(self.auxiliary_vocabulary.get(w, UNK_A) for w in self.tokenizer.split(x))
            idy = tuple(self.target_vocabulary.get(w, UNK_T) for w in self.tokenizer.split(y))
            return idx, idy, origin
        if origin == 0:
            return tuple(self.source_vocabulary.get(w, UNK_S) for w in self.tokenizer.split(sample))
        return tuple(self.auxiliary_vocabulary.get(w, UNK_A) for w in self.tokenizer.split(sample))

    def analyze_data(self, multi_data):
        data, aux_data = multi_data[0:2]
        counts_src = Counter()
        counts_tgt = Counter()
        for x, y in data:
            for w in self.tokenizer.split(x):
                counts_src[w] += 1
            for w in self.tokenizer.split(y):
                counts_tgt[w] += 1

        self.source_vocabulary_r = ['</s>', 'UNK', '<s>']
        self.source_vocabulary_r.extend(k for k, v in counts_src.most_common())
        self.source_vocabulary = {k: i for i, k in enumerate(self.source_vocabulary_r)}
        self.W_enc_emb.set_value(Random(len(self.source_vocabulary), self.embedding_size))

        # follows auxiliary data
        counts_aux = Counter()
        for x, y in aux_data:
            for w in self.tokenizer.split(x):
                counts_aux[w] += 1
            for w in self.tokenizer.split(y):
                counts_tgt[w] += 1
        self.auxiliary_vocabulary_r = ['</s>', 'UNK', '<s>']
        self.auxiliary_vocabulary_r.extend(k for k, v in counts_aux.most_common())
        self.auxiliary_vocabulary = {k: i for i, k in enumerate(self.auxiliary_vocabulary_r)}
        self.W_enc_emb_aux.set_value(Random(len(self.auxiliary_vocabulary), self.embedding_size))

        # follows target vocabulary
        self.target_vocabulary_r = ['</s>', 'UNK', '<s>']
        self.target_vocabulary_r.extend(k for k, v in counts_tgt.most_common())
        self.target_vocabulary = {k: i for i, k in enumerate(self.target_vocabulary_r)}
        self.W_dec_emb.set_value(Random(len(self.target_vocabulary), self.embedding_size))
        self.Wo2.set_value(Random(self.embedding_size, len(self.target_vocabulary)))
        self.bWo.set_value(Random(len(self.target_vocabulary)))
        return self._hyperparams


class BilingualSubWordModel(BilingualWordModel):
    tokenizer_cls = VowelTokenizer
