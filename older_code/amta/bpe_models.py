
from __future__ import print_function, division, absolute_import, unicode_literals
from .models import MonolingualSubWordModel, Encoder, Decoder
from subword_bpe.learn_bpe import main as learn_bpe
from subword_bpe.apply_bpe import BPE


class ModelCls(MonolingualSubWordModel):
    pass


class BPETokenizer(object):
    def __init__(self, codes=tuple(), symbols=30000):
        self.symbols = symbols
        codes_jo = ("{} {}".format(a, b) for a, b in codes)
        self.bpe = BPE(codes_jo)

    @staticmethod
    def _iterate_data(data):
        for a, b in data:
            yield a
            yield b

    def learn(self, data):
        data_it = self._iterate_data(data)
        codes = [pair for pair in learn_bpe(data_it, self.symbols)]
        codes_jo = ("{} {}".format(a, b) for a, b in codes)
        self.bpe = BPE(codes_jo)
        return codes

    def split(self, text):
        tokens = self.bpe.segment(text).split()
        return tokens

    @staticmethod
    def join(tokens):
        return " ".join(tokens).replace("@@ ", "")

    def trimmings(self, string):
        return iter(())


class ModelBPE(MonolingualSubWordModel):
    tokenizer_cls = BPETokenizer

    def __init__(self, encoder_cls=Encoder, decoder_cls=Decoder, **kwargs):
        super(ModelBPE, self).__init__(encoder_cls, decoder_cls, **kwargs)
        self.hyperparameter('codes', [])
        self.tokenizer = self.tokenizer_cls(self.codes, self.maximum_vocabulary_size)

    def analyze_data(self, data):
        print("training BPE")
        codes = self.tokenizer.learn(data)
        del self.codes[:]
        self.codes.extend(codes)
        print("done training BPE")
        return super(ModelBPE, self).analyze_data(data)

    def from_dict(self, d):
        super(ModelBPE, self).from_dict(d)
        self.tokenizer = self.tokenizer_cls(self.codes)
