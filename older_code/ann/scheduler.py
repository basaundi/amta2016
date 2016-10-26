
from __future__ import print_function, division, absolute_import, unicode_literals
import pickle
from random import shuffle
import subprocess
import os
from .parameters import Parametric


class Section(object):
    def __init__(self, name, vocabulary_limit, minibatch_limit):
        self.name = name
        self.vocabulary_limit = vocabulary_limit
        self.minibatch_limit = minibatch_limit
        self.minibatches = []
        self.vocabulary_src = {0, 1}
        self.vocabulary_tgt = {0, 1}
        self.done = False

    def append(self, minibatch):
        self.minibatches.append(minibatch)
        for sample in minibatch:
            self.vocabulary_src |= set(sample[0])
            self.vocabulary_tgt |= set(sample[1])

        if len(self.minibatches) > self.minibatch_limit:
            for w in range(self.vocabulary_limit):
                if len(self.vocabulary_tgt) > self.vocabulary_limit:
                    break
                self.vocabulary_tgt |= {w}
            self.done = True

        if len(self.vocabulary_tgt) > self.vocabulary_limit:
            self.done = True

    def save(self):
        print("Saving section `{}'...".format(self.name))
        with open(self.name, 'wb') as fd:
            pickle.dump(self, fd)


class Bucket(object):
    def __init__(self, max_length, batch_size, prefix='train'):
        self.max_length = max_length
        self.batch_size = batch_size
        self.prefix = prefix
        self.total_count = 0

    def accepts(self, sample):
        for datum in sample:
            if isinstance(datum, (tuple, list)) and len(datum) > self.max_length:
                return False
        return True

    def append(self, sample):
        self._append_to_file(sample)
        self.total_count += 1

    def finish(self):
        fname = self._fname()
        with open(fname+".info", "wb") as fd:
            fd.write(b"{:10d}\n".format(self.total_count))

    def get_length(self):
        fname = self._fname()
        with open(fname + ".info", "rb") as fd:
            self.total_count = int(fd.read())
        return self.total_count

    def iterate(self, shuffle=True):
        fname = self._fname()
        fname = "{}.shuf".format(fname)
        if shuffle:
            self._shuffle()
        with open(fname, "rb") as fd:
            minibatch = []
            for line in fd:
                x, y = line.split(b"\t")
                sample = [int(i) for i in x.split()], [int(i) for i in y.split()]
                minibatch.append(sample)
                if len(minibatch) >= self.batch_size:
                    yield minibatch
                    minibatch = []

    def _fname(self):
        return "{}.bucket.{}".format(self.prefix, self.max_length)

    def remove_file(self):
        fname = self._fname()
        try:
            os.remove(fname)
        except:
            pass

    def _append_to_file(self, sample):
        fname = self._fname()
        with open(fname, "ab") as fd:
            x, y = sample
            fd.write(b"{}\t{}\n".format(" ".join(str(i) for i in x), " ".join(str(i) for i in y)))

    def _shuffle(self):
        fname = self._fname()
        with open(fname, "rb") as fin, open("{}.shuf".format(fname), "wb") as fout:
            p = subprocess.Popen(['/usr/bin/shuf'], stdin=fin, stdout=fout, stderr=subprocess.PIPE)
            p.wait()


class BatchScheduler(Parametric):
    def __init__(self, model, prefix='train', **kwargs):
        super(BatchScheduler, self).__init__(**kwargs)
        self.hyperparameter('bucket_step', 5, "Step for the different bucket sizes.")
        self.hyperparameter('max_sequence_length', 300, "Maximum length (in tokens) for the sentences.")
        self.hyperparameter('minibatch_words', 4000, "Maximum (desired) number of tokens in a minibatch.")
        self.hyperparameter('vocabulary_limit', 30000, "Maximum number of words in the vocabulary of a section.")
        self.hyperparameter('minibatch_limit', 3000, "Maximum number of minibatches in a section.")
        self.model = model
        self.prefix = prefix
        self.parameter('sections', set())
        self.parameter('section_schedule', [])
        self.parameter('current_section_name', None)
        self.parameter('minibatch_i', 0)
        self.parameter('total_samples', 0)
        self.current_section = None
        self._buckets = [Bucket(l, self.minibatch_words // l, self.prefix)
                         for l in range(self.bucket_step, self.max_sequence_length + 1, self.bucket_step)]

    def from_dict(self, d):
        super(BatchScheduler, self).from_dict(d)
        self._buckets = [Bucket(l, self.minibatch_words // l, self.prefix)
                         for l in range(self.bucket_step, self.max_sequence_length + 1, self.bucket_step)]

    def reset(self):
        self.sections = set()
        self.section_schedule = []
        self.current_section_name = None
        self.minibatch_i = 0
        self.total_samples =0

    def _put_in_bucket(self, sample):
        for bucket in self._buckets:
            if bucket.accepts(sample):
                return bucket.append(sample)
        return None

    def _new_section(self):
        name = '{}.{:03}'.format(self.prefix, len(self.sections))
        self.sections.add(name)
        return Section(name, self.vocabulary_limit, self.minibatch_limit)

    def prepare_data(self, data):
        self.reset()
        data.reset()

        for bucket in self._buckets:
            bucket.remove_file()

        # Fill buckets
        for sample in data:
            self.total_samples += 1
            prepared_sample = self.model.prepare_sample(sample)
            self._put_in_bucket(prepared_sample)

        for bucket in self._buckets:
            bucket.finish()

        self._reschedule()

    def get_next(self):
        if not self.section_schedule:
            self._reschedule()

        if self.minibatch_i == 0:
            self._load_next()
        elif not self.current_section:
            self._load_section(self.current_section_name)

        minibatch = self.current_section.minibatches[self.minibatch_i]
        self.minibatch_i += 1
        if self.minibatch_i >= len(self.current_section.minibatches):
            self.section_schedule.pop()
            self.minibatch_i = 0
        return zip(*minibatch)

    def _reschedule(self):
        print("Rescheduling samples...")
        lengths = [bucket.get_length() for bucket in self._buckets]
        maximum = max(lengths)
        ratios = [l / maximum for l in lengths]
        counts = [0] * len(ratios)
        iterators = [bucket.iterate() for bucket in self._buckets]

        self.sections = set()
        section = self._new_section()
        for n in range(1, maximum + 1):
            for i, it in enumerate(iterators):
                if ratios[i] > counts[i] / n:
                    counts[i] += 1
                    minibatch = next(it, None)
                    if not minibatch:
                        continue
                    section.append(minibatch)
                    if section.done:
                        section.save()
                        section = self._new_section()
        section.save()
        for it in iterators:
            it.close()

        self.minibatch_i = 0
        self.section_schedule = sorted(self.sections)
        shuffle(self.section_schedule)

    def _load_next(self):
        section_name = self.section_schedule[-1]
        self._load_section(section_name)

    def _load_section(self, section_name):
        print("Loading section `{}'.".format(section_name))
        with open(section_name) as fd:
            self.current_section = pickle.load(fd)
        if self.minibatch_i == 0:
            print("Shuffling minibatches.")
            shuffle(self.current_section.minibatches)
            self.current_section.save()
        self.current_section_name = section_name

    def _set_vocabulary(self):
        pass


class BilingualBatchScheduler(Parametric):
    SRC, AUX = range(2)

    def __init__(self, model, **kwargs):
        super(BilingualBatchScheduler, self).__init__(**kwargs)
        self.parameter('src_scheduler', BatchScheduler(model, prefix='src'))
        self.parameter('aux_scheduler', BatchScheduler(model, prefix='aux'))
        self.parameter('auxiliar', False)

    def prepare_data(self, multi_data):
        data_src, data_aux = multi_data[0:2]
        self.src_scheduler.prepare_data(data_src)
        self.aux_scheduler.prepare_data(data_aux)

    def get_next(self):
        self.auxiliar = not self.auxiliar
        if self.auxiliar:
            return tuple(self.aux_scheduler.get_next()) + (self.AUX,)
        return tuple(self.src_scheduler.get_next()) + (self.SRC,)
