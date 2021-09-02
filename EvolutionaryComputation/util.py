import numpy as np
from sklearn.metrics import log_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization, Flatten, Dense, \
    Activation
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.distance import pdist
import tensorflow as tf
import sys
import time
import pandas as pd
import copy
import gc
import os
import pickle
import psutil  # psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def selu(x):
    alpha = 1.6733
    scale = 1.0507
    #signal = np.clip(x, -100, 100)
    return np.where(x > 0, scale*x, alpha*scale*(np.exp(x)-1))

def relu(x):
    return np.maximum(x, 0)


def elu(x):
    return np.where(x > 0, x, np.exp(x)-1)


def leaky_relu(x):
    return np.where(x > 0, x, x*0.2)


def tanh(x):
    signal = np.clip(x, -100, 100)
    return np.tanh(signal)


def sigmoid(x):
    signal = np.clip(x, -100, 100)
    return 1 / (1 + np.exp(-signal))


def unit(x):
    return np.heaviside(x, 0)


def softmax(x):
    signal = np.clip(x, -100, 100)
    return np.exp(signal - np.max(signal)) / np.sum(np.exp(signal - np.max(signal)))


def gaussian(x):
    signal = np.clip(x, -100, 100)
    return np.exp(- (signal ** 2))


def purlin(x):
    return x


def mse_error(actual, prediction):
    return np.sum(np.square(np.subtract(actual, prediction)).mean())


def r_2_error(actual, prediction):
    ss_resid = np.sum(np.square(np.subtract(actual, prediction)))
    ss_total = np.sum(np.square(np.subtract(actual, np.mean(actual))))
    return 1 - (float(ss_resid))/ss_total


def mae_error(actual, prediction):
    return np.sum(np.abs(np.subtract(actual, prediction)).mean())


def cross_entropy_error(actual, prediction):
    return log_loss(actual, prediction)