#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, chisquare


NUM_CATEGORIES = 10
SAMPLE_SIZE = 100
SIGNIFICANCE_LEVEL = 0.05
N = 1000

if __name__ == '__main__':
    # task 4a
    print("Task 4a:")
    sample = np.random.uniform(size=SAMPLE_SIZE)
    f_obs = np.zeros(NUM_CATEGORIES, dtype=int)
    for x in sample:
        f_obs[int(x / (1. / NUM_CATEGORIES))] += 1
    print('Observed frequences in the sample: {}'.format(f_obs))
    
    cs_test = chisquare(f_obs)
    ks_test = kstest(sample, 'uniform')
    print('P-value of the chi-square test: {}, within {} significance level the null hypothesis is {}'.format(
        cs_test.pvalue, SIGNIFICANCE_LEVEL, 'rejected' if cs_test.pvalue <= SIGNIFICANCE_LEVEL else 'not rejected'))
    print('P-value of the K-S test: {}, within {} significance level the null hypothesis is {}'.format(
        ks_test.pvalue, SIGNIFICANCE_LEVEL, 'rejected' if ks_test.pvalue <= SIGNIFICANCE_LEVEL else 'not rejected'))

    # task 4b
    alphas = np.arange(0., 1.05, step=.05)
    freq_rej = np.zeros(alphas.shape)
    for _ in range(N):
        sample = np.random.uniform(size=SAMPLE_SIZE)
        ks_test = kstest(sample, 'uniform')
        for i, alpha in enumerate(alphas):
            if ks_test.pvalue <= alpha:
                freq_rej[i] += 1
    freq_rej /= N
    plt.plot(alphas, freq_rej)
    plt.xlabel('Significance level')
    plt.ylabel('Rejected samples frequency')
    plt.show()
