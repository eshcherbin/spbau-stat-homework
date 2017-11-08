#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm


SAMPLE_SIZE = 100
N = 1000

def type2_errors(low, high):
    xs = np.arange(0., 1.01, step=.01)
    p_values_freq = np.zeros(xs.shape)
    for _ in range(N):
        sample = np.random.uniform(low=low, high=high, size=SAMPLE_SIZE)
        ks_test = kstest(sample, 'uniform')
        for i, x in enumerate(xs):
            if ks_test.pvalue >= x:
                p_values_freq[i] += 1
    p_values_freq /= N
    return xs, p_values_freq
    

if __name__ == '__main__':
    plt.figure(1)
    plt.subplot(121)

    # task 4a
    for low, high in [(0.05, 0.99), (0.05, 0.92), (0.1, 1.05), (0.5, 1.5)]:
        xs, p_values_freq = type2_errors(low, high)
        plt.plot(xs, p_values_freq, label=str((low, high)))
    plt.legend()
    plt.title('Task 4a')

    plt.subplot(122)

    # task 4b
    xs = np.arange(0., 1.01, step=.01)
    p_values_freq = np.zeros(xs.shape)
    for _ in range(N):
        sample = np.random.normal(size=SAMPLE_SIZE)
        mean, var = np.mean(sample), np.var(sample)
        ks_test = kstest(sample, lambda x: norm.cdf(x, loc=mean, scale=var))
        for i, x in enumerate(xs):
            if ks_test.pvalue <= x:
                p_values_freq[i] += 1
    p_values_freq /= N
    plt.plot(xs, p_values_freq)
    plt.title('Task 4b')

    plt.show()
