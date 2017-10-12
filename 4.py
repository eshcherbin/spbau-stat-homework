#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


def conf_interval(sample, level):
    mean, std = np.mean(sample), np.std(sample)
    n = len(sample)
    # percentile of Student's t-distribution
    tp = t.ppf((1 + level) / 2, n - 1)
    offset = std * tp / (n - 1)**.5
    return mean - offset, mean + offset

if __name__ == '__main__':
    plt.figure(1)
    plt.subplot(121)

    # a)
    loc = np.random.uniform(-10, 10)
    scale = np.random.uniform(0, 2)
    print('Generating 95% confidence intervals for mean={}, variance={} and '
          'n from 100 to 1000'.format(loc, scale))
    plot_data = []
    for n in range(100, 1001, 100):
        l, r = conf_interval(np.random.normal(loc, scale, size=n), 0.95)
        print('n: {}, interval: ({}, {}), width: {}'.format(n, l, r, r - l))
        plot_data.append((n, r - l))
    plt.scatter(*zip(*plot_data))
    plt.xlabel('Size')
    plt.ylabel('Width')
    plt.title('mean={:f}, variance={:f}'.format(loc, scale))

    print()
    plt.subplot(122)

    # b)
    loc = np.random.uniform(-10, 10)
    n = np.random.randint(100, 201)
    print('Generating 95% confidence intervals for mean={}, n={} and '
          'variance from 10 to 1'.format(loc, n))
    plot_data = []
    for scale in range(10, 0, -1):
        l, r = conf_interval(np.random.normal(loc, scale, size=n), 0.95)
        print('variance {}, interval: ({}, {}), width: {}'
              .format(scale, l, r, r - l))
        plot_data.append((scale, r - l))
    plt.scatter(*zip(*plot_data))
    plt.xlabel('Variance')
    plt.ylabel('Width')
    plt.title('mean={:f}, n={}'.format(loc, n))

    plt.show()
