from EvolutionaryComputation.util import *


# [-5.12, 5.12]
def shubert_function(x):
    if len(x.shape) == 1:
        t = [1, 2, 3, 4, 5]
        loc_sum = 0
        for i in range(0, len(t)):
            loc_sum += np.sum(t[i] * np.cos(x * (t[i] + 1) + t[i]))
        return loc_sum
    else:
        z = []
        t = [1, 2, 3, 4, 5]
        for row in x:
            loc_sum = 0
            for i in range(0, len(t)):
                loc_sum += np.sum(t[i] * np.cos(row * (t[i] + 1) + t[i]))
            z.append(loc_sum)
        return np.asarray(z)


# [-500, 500]
def schwefel_function(x):
    if len(x.shape) == 1:
        t = 418.9829 * x.shape[0]
        return t - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    else:
        z = []
        t = 418.9829 * x.shape[1]
        for row in x:
            z.append(t - np.sum(row * np.sin(np.sqrt(np.abs(row)))))
        return np.asarray(z)


# [-10, 10]
def levy_function(x):
    if len(x.shape) == 1:
        w = 1 + (x - 1) / 4
        t1 = np.power(np.sin(np.pi * w[0]), 2)
        t2 = np.power(x - 1, 2) * (1 + 10 * np.power(np.sin(np.pi * w + 1), 2))
        t3 = np.power(w[-1] - 1, 2) * (1 + np.power(np.sin(2 * np.pi * w[-1]), 2))
        return t1 + np.sum(t2 + t3)
    else:
        z = []
        for row in x:
            w = 1 + (row - 1) / 4
            t1 = np.power(np.sin(np.pi * w[0]), 2)
            t2 = np.power(row - 1, 2) * (1 + 10 * np.power(np.sin(np.pi * w + 1), 2))
            t3 = np.power(w[-1] - 1, 2) * (1 + np.power(np.sin(2 * np.pi * w[-1]), 2))
            z.append(t1 + np.sum(t2 + t3))
        return np.asarray(z)


# [-5, 10]
def zakharov_function(x):
    if len(x.shape) == 1:
        t1 = 0
        t2 = 0
        t3 = 0
        for i in range(0, len(x)):
            t1 += x[i] * x[i]
            t2 += 0.5 * (i + 1) * x[i]
            t3 += 0.5 * (i + 1) * x[i]
        return t1 + t2 * t2 + t3 * t3 * t3 * t3
    else:
        z = []
        for row in x:
            t1 = 0
            t2 = 0
            t3 = 0
            for i in range(0, len(row)):
                t1 += row[i] * row[i]
                t2 += 0.5 * (i + 1) * row[i]
                t3 += 0.5 * (i + 1) * row[i]
            z.append(t1 + t2 * t2 + t3 * t3 * t3 * t3)
        return np.asarray(z)


# [-5, 10]
def rosenbrock_function(x):
    if len(x.shape) == 1:
        loc_sum = 0
        for i in range(0, len(x) - 1):
            loc_sum += 100 * np.power((x[i + 1] - x[i] * x[i]), 2) + (x[i] - 1) * (x[i] - 1)
        return loc_sum
    else:
        z = []
        for row in x:
            loc_sum = 0
            for i in range(0, len(row) - 1):
                loc_sum += 100 * np.power((row[i + 1] - row[i] * row[i]), 2) + (row[i] - 1) * (row[i] - 1)
            z.append(loc_sum)
        return np.asarray(z)


# [-5, 5]
def styblinski_tang_function(x):
    if len(x.shape) == 1:
        return 0.5 * np.sum(np.power(x, 4) - 16 * np.power(x, 2) + 5 * x)
    else:
        z = []
        for row in x:
            z.append(0.5 * np.sum(np.power(row, 4) - 16 * np.power(row, 2) + 5 * row))
        return np.asarray(z)


def eggholder_function(x):  # [512,-512] min at -959.6407
    if len(x.shape) == 1:
        return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(
        np.sqrt(np.abs(x[0] - (x[1] + 47))))
    else:
        return -(x[:, 1] + 47) * np.sin(np.sqrt(np.abs(x[:, 0] / 2 + (x[:, 1] + 47)))) - x[:, 0] * np.sin(
            np.sqrt(np.abs(x[:, 0] - (x[:, 1] + 47))))

def rastrigin_function(x):  # [5.12,-5.12] vector at 0 vector
    if len(x.shape) == 1:
        d = len(x)
        return 10 * d + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x), axis=1)
    else:
        d = np.shape(x)[1]
        return 10 * d + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x), axis=1)


# Bowl Shaped

def sphere_function(x):  # [5.12,-5.12] vector at 0 vector
    if len(x.shape) == 1:
        return np.sum(np.power(x, 2))
    else:
        t = [0] * len(x)
        for i in range(0, len(x)):
            t[i] = np.sum(x[i,] ** 2)
        return np.asarray(t)


def trid_function(x):  # [d^2,-d^2] where d = nvar min at -d(d+4)(d-1)/6
    if len(x.shape) == 1:
        c = len(x)
        glob_sum1 = np.zeros(shape=(1,))
        for j in range(0, c):
            sum1 = np.power(x[:, j] - 1, 2)
            glob_sum1 += sum1
        glob_sum2 = np.zeros(shape=(1,))
        for j in range(1, c):
            sum2 = x[:, j] * x[:, j - 1]
            glob_sum2 += sum2
        return glob_sum1 - glob_sum2
    else:
        n, c = np.shape(x)
        glob_sum1 = np.zeros(shape=(n,))
        for j in range(0, c):
            sum1 = np.power(x[:, j] - 1, 2)
            glob_sum1 += sum1
        glob_sum2 = np.zeros(shape=(n,))
        for j in range(1, c):
            sum2 = x[:, j] * x[:, j - 1]
            glob_sum2 += sum2
        return glob_sum1 - glob_sum2


# Plate Shaped

def booth_function(x):  # [-10,10] min at x=0
    if len(x.shape) == 1:
        np.power(x[0] + 2 * x[1] - 7, 2) + np.power(2 * x[0] + x[1] - 5, 2)
    else:
        return np.power(x[:, 0] + 2 * x[:, 1] - 7, 2) + np.power(2 * x[:, 0] + x[:, 1] - 5, 2)


def power_sum_4d_function(x):  # [0, 4] for d=4 min at ???
    if len(x.shape) == 1:
        b = [8, 18, 44, 114]
        c = len(x)
        glob_sum1 = np.zeros(shape=(1,))
        for i in range(1, 5):
            glob_sum2 = np.zeros(shape=(1,))
            for j in range(0, 4):
                sum2 = np.power(x[:, j], i)
                glob_sum2 += sum2
            glob_sum2 = glob_sum2 - b[i - 1]
            glob_sum1 += np.power(glob_sum2, 2)
        return glob_sum1
    else:
        b = [8, 18, 44, 114]
        n, c = np.shape(x)
        glob_sum1 = np.zeros(shape=(n,))
        for i in range(1, 5):
            glob_sum2 = np.zeros(shape=(n,))
            for j in range(0, 4):
                sum2 = np.power(x[:, j], i)
                glob_sum2 += sum2
            glob_sum2 = glob_sum2 - b[i - 1]
            glob_sum1 += np.power(glob_sum2, 2)
        return glob_sum1


# Valley
def sixhump_camelback_function(x):  # x1=[-3,3], x2=[-2,2] min at -1.0316
    if len(x.shape) == 1:
        return (4 - 2.1 * np.power(x[0], 2) + np.power(x[0], 4) / 3) * np.power(x[0], 2) + x[0] * x[1] + (
                       -4 + 4 * np.power(x[1], 2)) * np.power(x[1], 2)
    else:
        return (4 - 2.1 * np.power(x[:, 0], 2) + np.power(x[:, 0], 4) / 3) * np.power(x[:, 0], 2) + x[:, 0] * x[:,
                                                                                                              1] + (
                       -4 + 4 * np.power(x[:, 1], 2)) * np.power(x[:, 1], 2)



# Steep Ridges/Drops
def easom_function(x):  # [100,-100] min at -1
    if len(x.shape) == 1:
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-np.power(x[0] - np.pi, 2) - np.power(x[1] - np.pi, 2))
    else:
        return -np.cos(x[:, 0]) * np.cos(x[:, 1]) * np.exp(-np.power(x[:, 0] - np.pi, 2) - np.power(x[:, 1] - np.pi, 2))


# OTHER

A = np.asarray([[10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14]])
P = pow(10, -4) * np.asarray([[1312, 1696, 5569, 124, 8283, 5886],
                              [2329, 4135, 8307, 3736, 1004, 9991],
                              [2348, 1451, 3522, 2883, 3047, 6650],
                              [4047, 8828, 8732, 5743, 1091, 381]])
alpha = np.asarray([1, 1.2, 3, 3.2])


def hartman_6D_function(x):  # [0,1] min at -3.32237
    if len(x.shape) == 1:
        value = 0
        for i in range(0, 4):
            val2 = 0
            for j in range(0, 6):
                val2 += A[i, j] * np.power(x[j] - P[i, j], 2)
            value += alpha[i] * np.exp(-val2)
        return -value
    else:
        value = 0
        for i in range(0, 4):
            val2 = 0
            for j in range(0, 6):
                val2 += A[i, j] * np.power(x[:, j] - P[i, j], 2)
            value += alpha[i] * np.exp(-val2)
        return -value


# fitness function
def pressure_vessel(x):
    if len(x.shape) == 1:
        x1 = np.round(x[0] / 0.0625, 0) * 0.0625
        x2 = np.round(x[1] / 0.0625, 0) * 0.0625
        x3 = x[2]
        x4 = x[3]
    else:
        x1 = np.round(x[:, 0] / 0.0625, 0) * 0.0625
        x2 = np.round(x[:, 1] / 0.0625, 0) * 0.0625
        x3 = x[:, 2]
        x4 = x[:, 3]
    return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 * x3 + 3.1661 * x1 * x1 * x4 + 19.84 * x1 * x1 * x3


# constraints
def constraints_pressure_vessel(x):
    if len(x.shape) == 1:
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
    else:
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
    g1 = -x1 + 0.0193 * x3

    g2 = -x2 + 0.00954 * x3

    g3 = -np.pi * x3 * x3 * x4 - (4 * np.pi / 3) * x3 * x3 * x3 + 1296000

    g4 = x4 - 240

    return np.asarray([g1, g2, g3, g4]).T
