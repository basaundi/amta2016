
from __future__ import print_function, division, absolute_import, unicode_literals
from io import open
import os.path
from .parameters import Parametric


class Data(Parametric):
    def __init__(self, path="train.set", start=0, stop=None, step=1, **kwargs):
        super(Data, self).__init__(**kwargs)
        if step != 1:
            raise NotImplementedError("step not implemented")
        self.parameter('path', os.path.abspath(path))
        self.parameter('offset', 0)
        self.parameter('start', start if start else 0)
        self.parameter('stop', stop)
        self.parameter('step', step if step else 1)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = self.start + (key.start or 0)
            stop = self.start + key.stop if key.stop else self.stop
            step = self.step * (key.step or 1)
            return Data(self.path, start, stop, step)
        else:
            raise TypeError("argument must be a slice")

    def __iter__(self):
        with open(self.path, 'r', encoding='utf-8') as fd:
            self.seek(fd, self.offset)
            # yield from zip(sfd, tfd)
            for offset, line in enumerate(fd):
                if self.stop and (self.start + offset >= self.stop):
                    break
                self.offset = offset
                if line.strip():
                    sample = self.preprocess(line)
                    if sample:
                        yield sample
            self.reset()

    def __len__(self):
        if self.stop:
            return 1 + ((self.stop - (abs(self.step)//self.step) - self.start) // self.step)
        with open(self.path, 'r', encoding='utf-8') as fd:
            self.seek(fd, self.start)
            for i, line in enumerate(fd):
                pass
            return i + 1

    def seek(self, fd, offset):
        self.offset = offset
        lineno = self.start + offset
        print("fast seeking to line {}".format(lineno))
        fd.seek(0)
        for i in range(lineno):
            next(fd)

    def reset(self):
        self.offset = 0

    def preprocess(self, line):
        t = line.strip().split('\t')
        if len(t) == 2 and t[0] and t[1]:
            return t
        return None


class MultiData(Data):
    def __init__(self, *args, **kwargs):
        super(MultiData, self).__init__(**kwargs)
        self.parameter('data', tuple(args))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def reset(self):
        for datum in self.data:
            datum.reset()
