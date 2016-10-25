#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals
import re
import unicodedata


class SplitTokenizer(object):
    def split(self, text):
        return text.lower().split()

    def join(self, tokens):
        return " ".join(tokens)


def memoize(fn):
    cache = {}

    def fn2(self, a):
        if a not in cache:
            cache[a] = fn(self, a)
        return cache[a]
    return fn2


class VowelTokenizer(object):
    basic_vowels = ("A", "E", "I", "O", "U", "Y", "AE")
    vowel_name_regex = re.compile(r"^LATIN SMALL LETTER ({})\b".format("|".join(basic_vowels)))
    WORD_BOUNDARY = u'\u241F'
    NUCLEUS_BOUNDARY = u"\u241D"

    @memoize
    def _is_vowel(self, char):
        try:
            is_vowel = self.vowel_name_regex.match(unicodedata.name(char))
        except:
            is_vowel = False
        return is_vowel

    @memoize
    def _is_letter(self, c):
        return unicodedata.category(c) in ("Ll", "Lu")

    @memoize
    def _is_digit(self, c):
        return unicodedata.category(c) == "Nd"

    def _split_syllables(self, letters):
        consonantal = True
        onset = []
        nucleus = []
        coda = []
        for c in letters:
            if self._is_vowel(c):
                consonantal = False
                if coda:
                    yield "".join(onset + [self.NUCLEUS_BOUNDARY, ] + nucleus +
                                  [self.NUCLEUS_BOUNDARY, ] + coda)
                    onset = coda
                    nucleus = []
                    coda = []
                nucleus.append(c)
            else:
                (coda if nucleus else onset).append(c)
        if nucleus or consonantal:
            yield "".join(onset + [self.NUCLEUS_BOUNDARY, ] + nucleus +
                          [self.NUCLEUS_BOUNDARY, ] + coda)

    def _split_word(self, word):
        group = None
        groups = []
        last_category = None
        for c in word:
            cat = unicodedata.category(c)[0]
            if cat != last_category:
                group = []
                groups.append((group, cat))
                last_category = cat
            group.append(c)

        for i, (group, cat) in enumerate(groups):
            c0 = group[0]
            if self._is_letter(c0):
                sws = list(self._split_syllables(group))
            elif self._is_digit(c0):
                sws = ["".join(group)]
            else:
                sws = ["".join(group)]

            if len(groups) > 1 and cat in ("L", "N"):
                if i != 0 and not groups[i - 1][1] in ("L", "N"):
                    sws[0] = self.WORD_BOUNDARY + sws[0]
                if group is not groups[-1][0] and not groups[i + 1][1] in ("L", "N"):
                    sws[-1] += self.WORD_BOUNDARY

            for sw in sws:
                yield sw

    def split(self, sentence):
        words = sentence.lower().split()
        res = []
        for word in words:
            tokens = list(self._split_word(word))
            tokens[0] = self.WORD_BOUNDARY + tokens[0]
            tokens[-1] += self.WORD_BOUNDARY
            res.extend(tokens)
        return res

    def join(self, tokens):
        WB = self.WORD_BOUNDARY
        NB = self.NUCLEUS_BOUNDARY
        SP = "\u2420"
        text = SP.join(tokens)
        text = re.sub(r'{}UNK{}'.format(SP, SP), r'{}{}UNK{}{}'.format(SP, WB, WB, SP), text)
        text = re.sub(r'^{}|{}$'.format(WB, WB), '', text)
        text = re.sub(r'([^{}{}{}]+){}\1'.format(NB, WB, SP, SP), r'\1', text)
        text = re.sub(r'{}{}{}'.format(WB, SP, WB), ' ', text)
        text = re.sub(r'{}|{}|{}'.format(SP, NB, WB), '', text)
        return text

    def trimmings(self, string):
        if string.startswith(self.WORD_BOUNDARY):
            if not string.endswith(self.WORD_BOUNDARY):
                for i, c in enumerate(string):
                    if c == self.NUCLEUS_BOUNDARY:
                        break
                    yield string[i + 1:]
        elif string.endswith(self.WORD_BOUNDARY):
            for i, c in enumerate(string[::-1]):
                if c == self.NUCLEUS_BOUNDARY:
                    break
                yield string[:-i - 1]
        else:
            lcount = 0
            rcount = 0
            string_it = iter(string)
            for c in string_it:
                if c == self.NUCLEUS_BOUNDARY:
                    break
                lcount += 1
            for c in string_it:
                if c == self.NUCLEUS_BOUNDARY:
                    continue
                rcount += 1

            rcount = len(string) - rcount - lcount + 1
            lskip = 0
            rskip = 0
            while lcount or rcount:
                if lcount > rcount:
                    lcount -= 1
                    lskip += 1
                else:
                    rcount -= 1
                    rskip += 1
                yield string[lskip:-rskip]


if __name__ == "__main__":
    sample = "This is a brief example of what polychromatic tokenizers-in-1998 can do with C19."
    tokenizer = VowelTokenizer()
    toks = tokenizer.split(sample)
    print(sample.lower())
    print(tokenizer.join(toks))
    print(toks)
