# V-COMET method

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from itertools import product


# min-max normalization
def minmax_normalization(X, criteria_type):
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    ind_profit = np.where(criteria_type == 1)
    ind_cost = np.where(criteria_type == -1)

    x_norm[:, ind_profit] = (X[:, ind_profit] - np.amin(X[:, ind_profit], axis = 0)
                             ) / (np.amax(X[:, ind_profit], axis = 0) - np.amin(X[:, ind_profit], axis = 0))

    x_norm[:, ind_cost] = (np.amax(X[:, ind_cost], axis = 0) - X[:, ind_cost]
                           ) / (np.amax(X[:, ind_cost], axis = 0) - np.amin(X[:, ind_cost], axis = 0))

    return x_norm


# max normalization
def max_normalization(X, criteria_type):
    maximes = np.amax(X, axis=0)
    ind = np.where(criteria_type == -1)
    X = X/maximes
    X[:,ind] = 1-X[:,ind]
    return X


# sum normalization
def sum_normalization(X, criteria_type):
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    ind_profit = np.where(criteria_type == 1)
    ind_cost = np.where(criteria_type == -1)

    x_norm[:, ind_profit] = X[:, ind_profit] / np.sum(X[:, ind_profit], axis = 0)

    x_norm[:, ind_cost] = (1 / X[:, ind_cost]) / np.sum((1 / X[:, ind_cost]), axis = 0)

    return x_norm


# equal weighting
def mean_weighting(X):
    N = np.shape(X)[1]
    return np.ones(N) / N


# entropy weighting
def entropy_weighting(X):
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    pij = sum_normalization(X, criteria_type)

    Ej = np.zeros(np.shape(pij)[1])
    for el in range(0, np.shape(pij)[1]):
        if np.any(pij[:, el] == 0):
            Ej[el] = 0
        else:
            Ej[el] = - np.sum(pij[:, el] * np.log(pij[:, el]))

    Ej = Ej / np.log(X.shape[0])

    wagi = (1 - Ej) / (np.sum(1 - Ej))
    return wagi


# standard deviation weighting
def std_weighting(X):
    stdv = np.std(X, axis = 0)
    return stdv / np.sum(stdv)


# CRITIC weighting
def critic_weighting(X):
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    x_norm = minmax_normalization(X, criteria_type)
    std = np.std(x_norm, axis = 0)
    n = np.shape(x_norm)[1]
    correlations = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            correlations[i, j], _ = pearsonr(x_norm[:, i], x_norm[:, j])

    difference = 1 - correlations
    suma = np.sum(difference, axis = 0)
    C = std * suma
    w = C / (np.sum(C, axis = 0))
    return w


# VIKOR method
def vikor(matrix, weights, criteria_types):
    v = 0.5
    nmatrix = minmax_normalization(matrix, criteria_types)
    fstar = np.amax(nmatrix, axis = 0)
    fminus = np.amin(nmatrix, axis = 0)
    weighted_matrix = weights * ((fstar - nmatrix) / (fstar - fminus))
    S = np.sum(weighted_matrix, axis = 1)
    R = np.amax(weighted_matrix, axis = 1)
    Sstar = np.min(S)
    Sminus = np.max(S)
    Rstar = np.min(R)
    Rminus = np.max(R)
    Q = v * (S - Sstar) / (Sminus - Sstar) + (1 - v) * (R - Rstar) / (Rminus - Rstar)
    return Q


# procedures for COMET method
def tfn(x, a, m, b):
    if x < a or x > b:
        return 0
    elif a <= x < m:
        return (x-a) / (m-a)
    elif m < x <= b:
        return (b-x) / (b-m)
    elif x == m:
        return 1


def evaluate_alternatives(C, x, ind):
    if ind == 0:
        return tfn(x, C[ind], C[ind], C[ind + 1])
    elif ind == len(C) - 1:
        return tfn(x, C[ind - 1], C[ind], C[ind])
    else:
        return tfn(x, C[ind - 1], C[ind], C[ind + 1])


#create characteristic values
def get_characteristic_values(matrix):
    cv = np.zeros((matrix.shape[1], 3))
    for j in range(matrix.shape[1]):
        cv[j, 0] = np.min(matrix[:, j])
        cv[j, 1] = np.mean(matrix[:, j])
        cv[j, 2] = np.max(matrix[:, j])
    return cv


#comet algorithm
def COMET(matrix, weights, criteria_types):
    # generate characteristic values
    cv = get_characteristic_values(matrix)
    # generate matrix with COs
    # cartesian product of characteristic values for all criteria
    co = product(*cv)
    co = np.array(list(co))
    # calculate vector SJ using VIKOR method
    sj = vikor(co, weights, criteria_types)

    # calculate vector P
    k = np.unique(sj).shape[0]
    p = np.zeros(sj.shape[0], dtype=float)

    for i in range(1, k):
        ind = sj == np.min(sj)
        p[ind] = (k - i) / (k - 1)
        sj[ind] = 1

    # inference and obtaining preference for alternatives
    preferences = []

    for i in range(len(matrix)):
        alt = matrix[i, :]
        W = []
        score = 0

        for i in range(len(p)):
            for j in range(len(co[i])):
                for index in range(len(cv[j])):
                    if cv[j][index] == co[i][j]:
                        ind = index
                        break
                W.append(evaluate_alternatives(cv[j], alt[j], ind))
            score += np.product(W) * p[i]
            W = []
        preferences.append(score)
    preferences = np.asarray(preferences)

    rankingPrep = np.argsort(-preferences)
    rank = np.argsort(rankingPrep) + 1

    return preferences, rank


# main
# choose year
year = '2019'
# choose type of dataset: absolute or relative
dataset = 'relative'
file = 'RES_EU_' + year + '_' + dataset + '.csv'
data = pd.read_csv(file)

list_alt_names = []
for i in range(1, len(data) + 1):
    list_alt_names.append(r'$A_{' + str(i) + '}$')

# choose weighting method
weight_type = 'equal'
# model hierarchization by decomposition
modules = [[1,2,3,4,5], [6,7,8,9], [10,11,12], [13,14,15]]

df_writer = pd.DataFrame()
df_writer['Ai'] = list_alt_names
scores = pd.DataFrame()
for el, m in enumerate(modules):
    df_matrix = data.iloc[:, m]
    matrix = df_matrix.to_numpy()

    # in this problem there are only profit criteria
    criteria_types = np.ones(np.shape(matrix)[1])

    if weight_type == 'equal':
        weights = mean_weighting(matrix)
    elif weight_type == 'entropy':
        weights = entropy_weighting(matrix)
    elif weight_type == 'std':
        weights = std_weighting(matrix)
    elif weight_type == 'CRITIC':
        weights = critic_weighting(matrix)

    pref, _ = COMET(matrix, weights, criteria_types)
    scores['P' + str(el + 1)] = pref

#
# outputs of modules are inputs for next module
matrix = scores.to_numpy()
# in this problem there are only profit criteria
criteria_types = np.ones(np.shape(matrix)[1])

if weight_type == 'equal':
    weights = mean_weighting(matrix)
elif weight_type == 'entropy':
    weights = entropy_weighting(matrix)
elif weight_type == 'std':
    weights = std_weighting(matrix)
elif weight_type == 'CRITIC':
    weights = critic_weighting(matrix)

pref, rank = COMET(matrix, weights, criteria_types)

df_writer['V-COMET pref'] = pref
df_writer['V-COMET rank'] = rank

df_writer = df_writer.set_index('Ai')
df_writer.to_csv('VCOMET_RES_' + dataset + '_' + year + '.csv')