#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)
import os.path
import numpy
import matplotlib

try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


def main(validation, tail, *logs):
    fig = plt.figure()
    plt.style.use('classic')
    ax1 = fig.add_subplot(111)
    if validation:
        ax2 = plt.twinx()
        ax2.yaxis.tick_right()
    if len(logs) == 1:
        dst_file = os.path.splitext(logs[0])[0] + '.png'
    else:
        dst_file = os.path.dirname(logs[0]) + '/joint_plot.png'

    prefix = 'last_cost'
    for arg_data in logs:
        with open(arg_data) as fd:
            header = next(fd).strip()
            data = [map(float, line.strip().split(',')) for line in fd if "None" not in line]
        header = header.split(',')
        cost_colums = {n_col: col for n_col, col in enumerate(header)
                       if col.startswith(prefix)}
        all_values = numpy.array(data)
        if tail:
            all_values = all_values[-tail:, :]
        for n_col, col in cost_colums.items():
            values = all_values[:, n_col]
            n = values.shape[0] // 500 if values.shape[0] > 500 else 1
            values = numpy.convolve(values, numpy.ones((n,))/n, mode=b'valid')
            valid_index_last = (len(all_values) - len(values))
            valid_index_first = valid_index_last // 2 + valid_index_last % 2
            valid_index_last //= 2
            valid_index_last = -valid_index_last or None
            ax1.plot(all_values[valid_index_first:valid_index_last, 0],
                     values, label="{} {}".format(
                        os.path.basename(os.path.dirname(arg_data)),
                        col[len(prefix):]))

        if validation:
            with open(os.path.dirname(arg_data) + "/validation.csv") as fd:
                header = next(fd)
                validation_data = [tuple(map(float, line.strip().split('\t'))) for line in fd]
                validation_data = [tuple(0 for v in validation_data[0])] + validation_data
                if tail:
                    validation_data = [t for t in validation_data if t[0] > all_values[-1, 0] - tail]
                validation_data = numpy.array(validation_data)
            if validation_data.any():
                ax2.plot(validation_data[:, 0], validation_data[:, 2], "--",
                         label=os.path.basename(os.path.dirname(arg_data)))

    ax1.set_xlabel("number of iterations")
    ax1.set_ylabel("cost (NLL) per symbol")
    ax1.set_ylim((0, 4))
    if validation:
        ax2.set_ylabel("score on validation set (BLEU)")
        ax2.set_ylim((0, 20))
        ax2.legend(loc=2)
    ax1.legend(loc=3)
    plt.savefig(dst_file, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('--validation', type=bool, default=False)
    parser.add_argument('--tail', type=int, default=None)
    arguments = parser.parse_args()
    main(arguments.validation, arguments.tail, arguments.data)
