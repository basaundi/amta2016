
from __future__ import print_function, division, absolute_import, unicode_literals
from time import time
from datetime import datetime
import os.path

from multi_bleu import multi_bleu, tokenize_lower


class ScheduledTask(object):
    def __init__(self, every_x_seconds=None, every_n_iterations=None, every_x_training_seconds=None):
        assert sum(map(bool, (every_x_seconds, every_n_iterations, every_x_training_seconds))) == 1
        self.last_time = time()  # Don't save immediately
        self.triggered_times = 0
        self.frequency_time = every_x_seconds
        self.frequency_time_training = every_x_training_seconds
        self.frequency_iter = every_n_iterations
        self.observable = None

    def link(self, observable):
        assert not self.observable, 'Can observe only one object'
        self.observable = observable

    def trigger(self):
        cls = type(self).__name__
        iters_now = self.observable.iterations_done
        training_now = self.observable.time_training
        if self.frequency_time:
            now = time()
            if (now - self.last_time) >= self.frequency_time:
                self.last_time = now
                self.observable.last_iter_triggered[cls] = iters_now
                self.observable.last_time_triggered[cls] = training_now
                self.do()
        elif self.frequency_time_training:
            last_time = self.observable.last_time_triggered.get(cls, 0)
            if (training_now - last_time) >= self.frequency_time_training:
                self.observable.last_iter_triggered[cls] = iters_now
                self.observable.last_time_triggered[cls] = training_now
                self.do()
        else:
            last_iter = self.observable.last_iter_triggered.get(cls, 0)
            if (iters_now - last_iter) >= self.frequency_iter:
                self.observable.last_iter_triggered[cls] = iters_now
                self.observable.last_time_triggered[cls] = training_now
                self.do()

    def reset(self):
        pass

    def do(self):
        raise NotImplementedError


class Saver(ScheduledTask):
    def do(self):
        print('triggered saving model...')
        self.observable.save(self.observable.path)
        print('triggered saving model done')


class Outputter(ScheduledTask):
    def do(self):
        if hasattr(self.observable, 'print_status'):
            self.observable.print_status()
        elif hasattr(self.observable, 'last_cost_src_tgt'):
            print("iteration {} ({!r}, {!r}) -> {} {} {} cost".format(
                self.observable.iterations_done + 1,
                self.observable.last_minibatch[1].shape,
                self.observable.last_minibatch[-1].shape,
                self.observable.last_cost_src_tgt,
                self.observable.last_cost_aux_tgt,
                self.observable.last_cost_src_src,))
        else:
            print("iteration {} ({!r}, {!r}) -> {} cost".format(
                self.observable.iterations_done + 1,
                self.observable.last_minibatch[1].shape,
                self.observable.last_minibatch[-1].shape,
                self.observable.last_cost))


class ProgressLogger(ScheduledTask):
    def __init__(self, file_name, fields, *args, **kwargs):
        super(ProgressLogger, self).__init__(*args, **kwargs)
        self.filename = file_name
        self.fields = fields

    def do(self):
        with open(self.filename, 'a') as fd:
            fd.write(b",".join(str(getattr(self.observable, k)) for k in self.fields) + '\n')

    def reset(self):
        with open(self.filename, 'w') as fd:
            fd.write(b",".join(self.fields) + '\n')


class Validator(ScheduledTask):
    def __init__(self, file_name, data, sampler_cls, *args, **kwargs):
        super(Validator, self).__init__(*args, **kwargs)
        self.filename = os.path.expanduser(file_name)
        self.data = data
        self.sampler = None
        self.sampler_cls = sampler_cls
        self.candidates = tuple()
        self.core_words = 10000

    def do(self):
        print('triggered validator...')
        validations = getattr(self.observable, 'validations', {})
        best_validation = max(validations.values()) if validations else 0
        score = self._get_score()
        validations[self.observable.iterations_done] = score
        if score >= best_validation:
            print("Previous validations:")
            print(validations)
            print("Best score so far: {}. Previous was {}.".format(score, best_validation))
            self.observable.model.save(self.observable.model.path + '.best')
        self._log(score)

    def _get_score(self):
        print("[{}] start validator...".format(datetime.now()))
        if not self.sampler:
            print('building sampler...')
            self.sampler = self.sampler_cls(self.observable.model)
            print('preparing candidate word list...')
            self.data.reset()
            candidates = set(range(self.core_words))
            for sample in self.data:
                x, y = self.observable.model.prepare_sample(sample)
                candidates |= set(y)
            self.candidates = tuple(sorted(candidates))
        print("[{}] start validation...".format(datetime.now()))
        start_time = time()
        self.data.reset()
        ss, st = zip(*iter(self.data))
        original_candidate_list = self.observable.model.get_target_vocabulary()
        try:
            score, precisions, brevity_penalty, candidate_total_length, references_closes_length = \
                    multi_bleu(self._map(ss), (self._map_target(st),), tokenize_lower, 4)
        finally:
            self.observable.model.set_target_vocabulary(original_candidate_list)
        print("[{}] Took {} seconds".format(datetime.now(), time() - start_time))
        return score

    def _map(self, sentences):
        for i, sentence in enumerate(sentences):
            if (i % 100) == 0:
                print('validated {}/{} sentences.'.format(i, len(sentences)))
            yield self.sampler.sample(sentence)

    def _map_target(self, sentences):
        model = self.observable.model
        for i, sentence in enumerate(sentences):
            tokens = model.tokenizer.split(sentence)
            yield model.tokenizer.join(tokens)

    def _log(self, score):
        print('validation score: {}'.format(score))
        with open(self.filename, 'a') as fd:
            fd.write('{iter}\t{time}\t{score}\n'.format(
                iter=self.observable.iterations_done,
                time=self.observable.time_training,
                score=score))

    def reset(self):
        with open(self.filename, 'w') as fd:
            fd.write(b"iter,time,score\n")
