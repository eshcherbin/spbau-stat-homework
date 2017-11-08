#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


SAMPLE_SIZE = 50
N = 10000

if __name__ == '__main__':
    plt.subplot(121)

    # type I error distribution
    for rho in [-0.9, -0.5, 0, 0.5, 0.9]:
        xs = np.linspace(0, 1, 101)
        p_values_freq = np.zeros(xs.shape)
        for _ in range(N):
            sample1 = np.random.normal(size=SAMPLE_SIZE)
            sample_aux = np.random.normal(size=SAMPLE_SIZE)
            sample2 = rho * sample1 + (1 - rho**2)**0.5 * sample_aux
            ttest = ttest_ind(sample1, sample2)
            for i, x in enumerate(xs):
                if ttest.pvalue <= x:
                    p_values_freq[i] += 1
        p_values_freq /= N
        plt.plot(xs, p_values_freq, label='ρ = {}'.format(rho))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('cdf')
    plt.title('Type I error distribution')

    plt.subplot(122)

    # t statistic pdf 
    for rho in [-0.9, -0.5, 0, 0.5, 0.9]:
        xs = np.linspace(-4, 4, 50)
        delta = xs[1] - xs[0]
        t_stat_freq = np.zeros(xs.shape)
        for _ in range(N * 10):
            sample1 = np.random.normal(size=SAMPLE_SIZE)
            sample_aux = np.random.normal(size=SAMPLE_SIZE)
            sample2 = rho * sample1 + (1 - rho**2)**0.5 * sample_aux
            ttest = ttest_ind(sample1, sample2)
            for i, x in enumerate(xs):
                if ttest.statistic <= x:
                    t_stat_freq[i] += 1
                    break
        t_stat_freq /= N * 10 * delta
        plt.plot(xs, t_stat_freq, label='ρ = {}'.format(rho))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.title('t statistic pdf')

    plt.show()
