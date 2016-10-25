
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from .parameters import Parametric
from .zeano import scan, sigmoid, tanh, dot, zeros, floatX, stable_softmax_3


def Random(*args):
    return np.random.randn(*args).astype(floatX) * 0.01


class GRU(Parametric):
    def __init__(self, dim_word=500, dim=1000):
        super(GRU, self).__init__()
        self.parameter('Wi', Random(dim_word, dim))
        self.parameter('bi', Random(dim))
        self.parameter('Wr', Random(dim_word, dim))
        self.parameter('Wz', Random(dim_word, dim))
        self.parameter('Ui', Random(dim, dim))
        self.parameter('Ur', Random(dim, dim))
        self.parameter('Uz', Random(dim, dim))

    def update_state(self, mask, inp, rst, upd, prev):
        mask = mask[:, None]
        r = sigmoid(rst + dot(prev, self.Ur))
        z = sigmoid(upd + dot(prev, self.Uz))
        i = tanh(inp + dot((r * prev), self.Ui))
        h = z * prev + (1.0 - z) * i
        return mask * h + (1.0 - mask) * prev

    def apply(self, embs, mask):
        # shape of embs is (n_words_in_sentence, n_samples, emb_dimension)
        n_steps = embs.shape[0]
        n_samples = embs.shape[1]
        # note: np.dot(a, b) operates over the last axis of a and the second-to-last of b.
        inp = dot(embs, self.Wi) + self.bi
        rst = dot(embs, self.Wr)
        upd = dot(embs, self.Wz)
        previous = zeros((n_samples, self.Ur.shape[0]), dtype=floatX)
        outs, updts = scan(self.update_state, n_steps=n_steps,
                           sequences=(mask, inp, rst, upd),
                           outputs_info=(previous,))
        return outs


class GRUCond(GRU):
    def __init__(self, dim_word=500, dim=1000, ctx_dim=2000):
        super(GRUCond, self).__init__(dim_word, dim)
        self.parameter('W_init', Random(ctx_dim, dim))
        self.parameter('b_init', Random(dim))

        self.parameter('Wc', Random(dim_word, dim))  # prev_emb projection

        self.parameter('Wa', Random(dim, dim))       # state projection
        self.parameter('Ua', Random(ctx_dim, dim))   # ctx projection
        self.parameter('Va', Random(dim, 1))         # ctx+state+prev_emb combined softmax projection
        self.parameter('Ci', Random(ctx_dim, dim))   # context-projection-projection
        self.parameter('Cr', Random(ctx_dim, dim))   # "
        self.parameter('Cz', Random(ctx_dim, dim))   # "

    def init(self, ctx, x_mask):
        ctx_mean = (ctx * x_mask[:, :, None]).sum(0, keepdims=True) / x_mask.sum(0)[:, None]
        return tanh(dot(ctx_mean, self.W_init) + self.b_init)

    def context_vector(self, ctx, x_mask, state_i, eb, cti_embs_i, with_align=False):
        eia = dot(state_i[None, :, :], self.Wa)  # Projection of the state
        ei = dot(tanh(eia.repeat(eb.shape[0], axis=0) + eb + cti_embs_i), self.Va)
        ai = stable_softmax_3(ei.T).T[:, :, 0]
        ci = (ai[:, :, None] * ctx).sum(axis=0)
        if with_align:
            return ci, ai
        else:
            return ci

    def gate_projections(self, word_embs):
        upd_embs = dot(word_embs, self.Wz)
        rst_embs = dot(word_embs, self.Wr)
        inp_embs = dot(word_embs, self.Wi)
        cti_embs = dot(word_embs, self.Wc)  # Candidate context projection
        return upd_embs, rst_embs, inp_embs, cti_embs

    def gate_embeddings(self, upd_embs_i, rst_embs_i, inp_embs_i, ci):
        rst = rst_embs_i + dot(ci, self.Cr)
        upd = upd_embs_i + dot(ci, self.Cz)
        inp = inp_embs_i + dot(ci, self.Ci) + self.bi
        return inp, rst, upd

    def step(self, upd_embs_i, rst_embs_i, inp_embs_i, cti_embs_i, mask, prev, prev_ci, ctx, x_mask, eb):
        ci = self.context_vector(ctx, x_mask, prev, eb, cti_embs_i)
        rst, upd, inp = self.gate_embeddings(upd_embs_i, rst_embs_i, inp_embs_i, ci)
        h = self.update_state(mask, inp, rst, upd, prev)
        return h, ci

    def apply(self, ctx, x_mask, word_embs, mask, init_state):
        upd_embs, rst_embs, inp_embs, cti_embs = self.gate_projections(word_embs)
        eb = dot(ctx, self.Ua)  # Projection of ctx

        zero_ci = zeros((ctx.shape[1], self.Ci.shape[0]), dtype=floatX)
        n_steps = word_embs.shape[0]
        outs, updts = scan(self.step, n_steps=n_steps,
                           sequences=(upd_embs, rst_embs, inp_embs, cti_embs, mask),
                           outputs_info=(init_state[0], zero_ci),
                           non_sequences=(ctx, x_mask, eb))
        return outs
