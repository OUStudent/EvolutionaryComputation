import EvolutionaryComputation.GeneticAlgorithms as ga
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from scipy.stats import zscore
import pandas as pd
import numpy as np
boston = load_boston()
#print(boston.DESCR)

# number of observations
n = len(boston.data)

# create indices
ind = np.asarray(range(0, n))
# shuffle
np.random.shuffle(ind)

# percentages
train = 0.6
val = 0.8

# train, val, test indices
train_ind = ind[0:int(train*n)]  # train is 60% of data
val_ind = ind[int(train*n):int(val*n)]  # validation is 20% of data
test_ind = ind[int(val*n):n]  # test is 20% of data

x = pd.DataFrame(boston.data, columns = boston.feature_names)

y = np.asarray(boston['target'])

# convert to numpy array
x = np.asarray(x)
x = x[ind]
# scale between 0 and 1
mx = np.max(x, axis=0)
mn = np.min(x, axis=0)
x = (x-mn)/(mx-mn)

# scale between 0 and 1
mx = np.max(y, axis=0)
mn = np.min(y, axis=0)
y = (y-mn)/(mx-mn)
y = y[ind]

from sklearn.ensemble import RandomForestRegressor


def fitness_function(population, init_pop_print=False):
    fits = []
    for individual in population:
        if individual[0] <= 0.5:
            bootstrap = True
        else:
            bootstrap = False

        if individual[1] > 100:
            max_depth = None
        else:
            max_depth = int(np.round(individual[1], 0))

        if individual[2] <= 0.5:
            max_features = 'auto'
        else:
            max_features = 'sqrt'

        min_samples_leaf = int(np.round(individual[3], 0))
        min_samples_split = int(np.round(individual[4], 0))
        n_estimators = int(np.round(individual[5], 0))

        forest = RandomForestRegressor(bootstrap=bootstrap, max_depth=max_depth, max_features=max_features,
                                       min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                       n_estimators=n_estimators)
        kf = KFold(n_splits=3)
        loc_fit = []
        for train, test, in kf.split(x):
            forest.fit(x[train], y[train])
            loc_fit.append(np.sum(np.power(y[test]-forest.predict(x[test]), 2)))
        fits.append(np.sum(loc_fit))
        if init_pop_print:
            msg = 'Best: {}, Median: {}, Worst: {}'.format(np.min(fits), np.median(fits), np.max(fits))
            print(msg)
    return np.asarray(fits)


upper_bound = [1, 110, 1, 4, 10, 2000]
lower_bound = [0, 10, 0, 1, 2, 200]
size = 10

from scipy.optimize import fmin

def f(individual):
    if individual[0] <= 0.5:
        bootstrap = True
    else:
        bootstrap = False

    if individual[1] > 100:
        max_depth = None
    else:
        max_depth = np.maximum(10, int(np.round(individual[1], 0)))

    if individual[2] <= 0.5:
        max_features = 'auto'
    else:
        max_features = 'sqrt'

    min_samples_leaf = np.maximum(1, int(np.round(individual[3], 0)))
    min_samples_split = np.maximum(2, int(np.round(individual[4], 0)))
    n_estimators = np.maximum(10, int(np.round(individual[5], 0)))

    forest = RandomForestRegressor(bootstrap=bootstrap, max_depth=max_depth, max_features=max_features,
                                   min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                   n_estimators=n_estimators)
    kf = KFold(n_splits=3)
    loc_fit = []
    for train, test, in kf.split(x):
        forest.fit(x[train], y[train])
        loc_fit.append(np.sum(np.power(y[test] - forest.predict(x[test]), 2)))
    return np.sum(np.sum(loc_fit))

print(fmin(f, np.array([0.5, 75, 0.5, 2.5, 5, 1000])))

algorithm = ga.GenericUnconstrainedProblem(fitness_function=fitness_function, upper_bound=upper_bound,
                                           lower_bound=lower_bound, gen_size=size)

algorithm.evolve(max_iter=20, algorithm='differential', find_max=False, init_pop=100)
algorithm.plot()

