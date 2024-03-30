from utils import get_probability_distribution
import numpy as np


def implement_RRP(A, r):
    # Randomized Residual Projection Algorithm

    probability_distribution = get_probability_distribution(A)
    index_list = [i for i in range(0, len(probability_distribution))]
    if r.shape == (len(A), 1):
        r = r.reshape((len(A),))
    x = np.zeros(len(probability_distribution))                          # initialize x

    stage = 0
    while True:
        pick_random_index = np.random.choice(index_list, p=probability_distribution)
        a = A[:, pick_random_index]
        delta = np.inner(a, r) / np.inner(a, a)
        x[pick_random_index] = x[pick_random_index] + delta
        r = (r - (a * delta))

        stage += 1
        if stage >= 1000:
            break

    x = x.reshape((len(x), 1))
    r = r.reshape((len(r), 1))

    return (x, r)