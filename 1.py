#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math


SAMPLE_SIZE = 100
N = 1000

def moments_estimate_uniform(sample, k):
    return (sum(map(lambda x: x**k, sample)) * (k + 1) / len(sample))**(1. / k)


def moments_estimate_exponential(sample, k):
    return (sum(map(lambda x: x**k, sample)) / (len(sample) * math.gamma(k + 1)))**(1. / k)
    

def plot_estimate_variance(distribution, estimator):
    ks = range(1, 11);
    es = [np.mean([(estimator(distribution(size=SAMPLE_SIZE), k) - 1)**2 for _ in range(N)]) for k in ks]
    plt.plot(ks, es)
    plt.xlabel('k')
    plt.ylabel('E[(T-θ)^2]')


if __name__ == '__main__':
    plt.figure(1)
    plt.subplot(121)

    # uniform distribution
    plot_estimate_variance(np.random.uniform, moments_estimate_uniform)
    plt.title('Uniform distribution')

    plt.subplot(122)

    # exponential distribution
    plot_estimate_variance(np.random.exponential, moments_estimate_exponential)
    plt.title('Exponential distribution')


    plt.show()
