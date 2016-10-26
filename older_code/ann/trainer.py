
from __future__ import print_function, division, absolute_import, unicode_literals

import time
import os.path as osp
from .parameters import Parametric


class Trainer(Parametric):
    def __init__(self, model, data, optimizer_cls, scheduler_cls, observers=(), **kwargs):
        super(Trainer, self).__init__(**kwargs)

        self.hyperparameter('target_iterations', 3000000, "Number of iterations.")

        self.model = model
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        self.parameter('data', data)
        self.parameter('scheduler', scheduler_cls(model, **kwargs))
        self.parameter('optimizer', optimizer_cls(model, **kwargs))
        self.parameter('initialized', False)
        self.parameter('done', False)

        self.observers = []
        for observer in observers:
            self.observers.append(observer)
            observer.link(self)

        # stats
        self.parameter('validations', {})
        self.parameter('x_words_done', 0)
        self.parameter('y_words_done', 0)
        self.parameter('samples_done', 0)
        self.parameter('iterations_done', 0)
        self.parameter('epochs_done', 0)
        self.parameter('time_pulling_data', 0)
        self.parameter('time_training', 0)
        self.parameter('last_iter_triggered', {})
        self.parameter('last_time_triggered', {})
        self.last_cost = None
        self.last_minibatch = None
        self.last_minibatch_origin = 0

        self.initialized = False

        # private
        self._time_last_minibatch = 0
        self.cost_with_update = None

    def _pull_minibatch(self):
        minibatch = self.scheduler.get_next()
        return self.model.prepare_batch(*minibatch)

    def is_done(self):
        return self.iterations_done >= self.target_iterations

    def _update_state(self):
        pass

    def train(self):
        if not self.initialized:
            self.reset()
        self._get_cost_with_update()
        self._update_state()

        print('begin training...')
        try:
            while not self.is_done():
                try:
                    self._time_last_minibatch = time.time()
                    minibatch = self._pull_minibatch()
                    self._before_minibatch(minibatch)
                    self._process_minibatch(minibatch)
                    self._after_minibatch(minibatch)
                except StopIteration:
                    self.epochs_done += 1
                    continue
                self.iterations_done += 1
        except KeyboardInterrupt:
            print("Interrupt received.")
        finally:
            print("Saving progress...")
            self.save(self.path)
        print('training done')

    def _before_minibatch(self, minibatch):
        self.last_minibatch = minibatch
        self.time_pulling_data += time.time() - self._time_last_minibatch

    def _process_minibatch(self, minibatch):
        self.last_cost = self.cost_with_update(*minibatch)

    def _after_minibatch(self, minibatch):
        samples = minibatch[-1].shape[1]
        self.x_words_done += int(minibatch[1].sum()) - samples  # Discount EOS
        self.y_words_done += int(minibatch[3].sum()) - samples  # Discount EOS
        self.samples_done += samples
        self.time_training += time.time() - self._time_last_minibatch
        for observer in self.observers:
            observer.trigger()

    def save(self, path):
        super(Trainer, self).save(path)
        self.model.save(osp.join(osp.dirname(path), 'model.pkl'))

    def load(self, path):
        super(Trainer, self).load(path)
        self.model.load(osp.join(osp.dirname(path), 'model.pkl'))

    def prepare_data(self):
        print("preparing trainer data...")
        self.data.reset()
        self.scheduler.prepare_data(self.data)

    def reset(self):
        print("analyzing data...")
        self.data.reset()
        self.model.analyze_data(self.data)

        self.prepare_data()
        print("resetting trainer...")
        self.optimizer.reset()
        self.model.reset()
        self.initialized = True
        for observer in self.observers:
            observer.reset()
        self.save(self.path)

    def _get_cost_with_update(self):
        if self.cost_with_update:
            return
        print('compiling cost function...')
        self.cost_with_update = self.optimizer.get_cost_with_update()
        return self.cost_with_update
