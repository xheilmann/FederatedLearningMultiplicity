import math
from itertools import islice

# SPDX-License-Identifier: Apache-2.0
# Copyright : J.P. Morgan Chase & Co.
import pandas as pd
import numpy as np
from diffprivlib import tools as dp

import multiprocessing
from multiprocessing import Pool

def ambiguity(decisions, y):
    nsample = len(decisions[0])
    wrong_per_sample = np.sum(~(decisions == np.array(y)), axis=0)
    wrong_per_sample[wrong_per_sample>0] = 1
    return np.sum(wrong_per_sample) / nsample

def discrepancy(decisions, y):
    nsample = len(decisions[0])
    return np.max(np.sum(~(decisions == np.array(y)), axis=1) / nsample)

def disagreement_hat(decisions):
    # nmodel = decisions.shape[0]
    # nsample = decisions.shape[1]
    # nclass = scores.shape[2]
    mu = np.mean(decisions, axis=0)
    k = 4*np.multiply(mu, 1-mu)
    #counts, breaks = np.histogram(k, bins=1000, range=(0, 1))
    counts, breaks = dp.histogram(k, bins=1000, range=(0, 1), epsilon=0.1)
    return counts, k

## Score Based
def rashomon_capacity(scores):
    nmodel = len(scores)
    nsample =len(scores[0])
    nclass = len(scores[0][0])
    array = []
    for i in range(nsample):
        score_y = np.zeros((nmodel, nclass))
        for j in range(nmodel):
            for c in range(nclass):
                score_y[j, c] = sigmoid(scores[j][i][c])
                #score_y[j, 1] = sigmoid(scores[j][i][1])
        array.append(score_y)

    cores = multiprocessing.cpu_count() - 1
    #it = iter(range(nsample))
    #ln = list(iter(lambda: tuple(islice(it, 1)), ()))  # list of indices
    # compute in parallel
    with Pool(cores) as p:
        cvals = (p.map(blahut_arimoto, array))
    capacity = np.array([v[0] for v in cvals])
    counts, breaks= dp.histogram(capacity, bins=1000, range=(0, 1), epsilon=0.1)
    return counts, capacity

def viable_prediction_range(scores):
    vpr = scores.max(axis=0)-scores.min(axis=0)
   # print(vpr)
    #counts, breaks = np.histogram(vpr, bins=1000, range=(0,1))
    counts, breaks = dp.histogram(vpr, bins=1000, range=(0, 1), epsilon=0.1)
    return counts, vpr

def score_variance(scores):
    # nmodel, nsample, nclass = scores.shape[0], scores.shape[1], scores.shape[2]
    var = scores.var(axis=0)
    #counts, breaks = np.histogram(var, bins=100, range=(0, 1))
    counts, breaks = dp.histogram(var, bins=1000, range=(0, 1), epsilon=0.1)
    return counts, var

def blahut_arimoto(Pygw, log_base=2, epsilon=1e-12, max_iter=100):
    """
    Performs the Blahut-Arimoto algorithm to compute the channel capacity
    given a channel P_ygx.

    Parameters
    ----------
    Pygw: shape (m, c).
        transition matrix of the channel with m inputs and c outputs.
    log_base: int.
        base to compute the mutual information.
        log_base = 2: bits, log_base = e: nats, log_base = 10: dits.
    epsilon: float.
        error tolerance for the algorithm to stop the iterations.
    max_iter: int.
        number of maximal iteration.
    Returns
    -------
    Capacity: float.
        channel capacity, or the maximum information it can be transmitted
        given the input-output function.
    pw: array-like.
        array containing the discrete probability distribution for the input
        that maximizes the channel capacity.
    loop: int
        the number of iteration.
    resource: https://sites.ecse.rpi.edu/~pearlman/lec_notes/arimoto_2.pdf
    """
    ## check inputs
    # assert np.abs(Pygw.sum(axis=1).mean() - 1) < 1e-6
    # assert Pygw.shape[0] > 1

    m = Pygw.shape[0]
    c = Pygw.shape[1]
    Pw = np.ones((m)) / m
    for cnt in range(int(max_iter)):
        ## q = P_wgy
        q = (Pw * Pygw.T).T
        q = q / q.sum(axis=0)

        ## r = Pw
        r = np.prod(np.power(q, Pygw), axis=1)
        r = r / r.sum()

        ## stoppung criteria
        if np.sum((r - Pw) ** 2) / m < epsilon:
            break
        else:
            Pw = r

    ## compute capacity
    capacity = 0
    for i in range(m):
        for j in range(c):
            ## remove negative entries
            if r[i] > 0 and q[i, j] > 0:
                capacity += r[i] * Pygw[i, j] * np.log(q[i, j] / r[i])

    capacity = capacity / np.log(log_base)
    return capacity, r, cnt+1


def score_of_y_multi_model(scores, y):
    nmodel, nsample, nclass = len(scores), len(scores[0]), len(scores[0][0])
    score_y = np.zeros((nmodel, nsample,))
    for i in range(nmodel):
        for j in range(nsample):
            score_y[i, j] = sigmoid(scores[i][j][y[j]])
    return score_y

def sigmoid(x):
    return 1 / (1 + math.exp(-x))



# import math
# from itertools import islice

# # SPDX-License-Identifier: Apache-2.0
# # Copyright : J.P. Morgan Chase & Co.
# import pandas as pd
# import numpy as np
# from diffprivlib import tools as dp

# import multiprocessing
# from multiprocessing import Pool

# def ambiguity(decisions, y):
#     nsample = len(decisions[0])
#     wrong_per_sample = np.sum(~(decisions == np.array(y)), axis=0)
#     wrong_per_sample[wrong_per_sample>0] = 1
#     return np.sum(wrong_per_sample) / nsample

# def discrepancy(decisions, y):
#     nsample = len(decisions[0])
#     return np.max(np.sum(~(decisions == np.array(y)), axis=1) / nsample)

# def disagreement_hat(decisions):
#     # nmodel = decisions.shape[0]
#     # nsample = decisions.shape[1]
#     # nclass = scores.shape[2]
#     mu = np.mean(decisions, axis=0)
#     k = 4*np.multiply(mu, 1-mu)
#     #counts, breaks = np.histogram(k, bins=1000, range=(0, 1))
#     counts, breaks = dp.histogram(k, bins=1000, range=(0, 1), epsilon=0.1)
#     return counts, k

# ## Score Based
# def rashomon_capacity(scores):
#     nmodel = len(scores)
#     nsample =len(scores[0])
#     nclass = len(scores[0][0])
#     array = []
#     for i in range(nsample):
#         score_y = np.zeros((nmodel, nclass))
#         for j in range(nmodel):
#             score_y[j, 0] = sigmoid(scores[j][i][0])
#             score_y[j, 1] = sigmoid(scores[j][i][1])
#         array.append(score_y)

#     cores = multiprocessing.cpu_count() - 1
#     #it = iter(range(nsample))
#     #ln = list(iter(lambda: tuple(islice(it, 1)), ()))  # list of indices
#     # compute in parallel
#     with Pool(cores) as p:
#         cvals = (p.map(blahut_arimoto, array))
#     capacity = np.array([v[0] for v in cvals])
#     counts, breaks= dp.histogram(capacity, bins=1000, range=(0, 1), epsilon=0.1)
#     return counts, capacity

# def viable_prediction_range(scores):
#     vpr = scores.max(axis=0)-scores.min(axis=0)
#    # print(vpr)
#     #counts, breaks = np.histogram(vpr, bins=1000, range=(0,1))
#     counts, breaks = dp.histogram(vpr, bins=1000, range=(0, 2), epsilon=0.1)
#     return counts, vpr

# def score_variance(scores):
#     # nmodel, nsample, nclass = scores.shape[0], scores.shape[1], scores.shape[2]
#     var = scores.var(axis=0)
#     #counts, breaks = np.histogram(var, bins=100, range=(0, 1))
#     counts, breaks = dp.histogram(var, bins=1000, range=(0, 1), epsilon=0.1)
#     return counts, var

# def blahut_arimoto(Pygw, log_base=2, epsilon=1e-12, max_iter=100):
#     """
#     Performs the Blahut-Arimoto algorithm to compute the channel capacity
#     given a channel P_ygx.

#     Parameters
#     ----------
#     Pygw: shape (m, c).
#         transition matrix of the channel with m inputs and c outputs.
#     log_base: int.
#         base to compute the mutual information.
#         log_base = 2: bits, log_base = e: nats, log_base = 10: dits.
#     epsilon: float.
#         error tolerance for the algorithm to stop the iterations.
#     max_iter: int.
#         number of maximal iteration.
#     Returns
#     -------
#     Capacity: float.
#         channel capacity, or the maximum information it can be transmitted
#         given the input-output function.
#     pw: array-like.
#         array containing the discrete probability distribution for the input
#         that maximizes the channel capacity.
#     loop: int
#         the number of iteration.
#     resource: https://sites.ecse.rpi.edu/~pearlman/lec_notes/arimoto_2.pdf
#     """
#     ## check inputs
#     # assert np.abs(Pygw.sum(axis=1).mean() - 1) < 1e-6
#     # assert Pygw.shape[0] > 1

#     m = Pygw.shape[0]
#     c = Pygw.shape[1]
#     Pw = np.ones((m)) / m
#     for cnt in range(int(max_iter)):
#         ## q = P_wgy
#         q = (Pw * Pygw.T).T
#         q = q / q.sum(axis=0)

#         ## r = Pw
#         r = np.prod(np.power(q, Pygw), axis=1)
#         r = r / r.sum()

#         ## stoppung criteria
#         if np.sum((r - Pw) ** 2) / m < epsilon:
#             break
#         else:
#             Pw = r

#     ## compute capacity
#     capacity = 0
#     for i in range(m):
#         for j in range(c):
#             ## remove negative entries
#             if r[i] > 0 and q[i, j] > 0:
#                 capacity += r[i] * Pygw[i, j] * np.log(q[i, j] / r[i])

#     capacity = capacity / np.log(log_base)
#     return capacity, r, cnt+1


# def score_of_y_multi_model(scores, y):
#     nmodel, nsample, nclass = len(scores), len(scores[0]), len(scores[0][0])
#     score_y = np.zeros((nmodel, nsample,))
#     for i in range(nmodel):
#         for j in range(nsample):
#             score_y[i, j] = sigmoid(scores[i][j][y[j]])
#     return score_y

# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))