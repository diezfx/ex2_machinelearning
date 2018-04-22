#!/usr/bin/env python
# encoding: utf-8
"""
This is a mini demo of how to use numpy arrays and plot data.
NOTE: the operators + - * / are element wise operation. If you want
matrix multiplication use ‘‘dot‘‘ or ‘‘mdot‘‘!
"""
import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D  # 3D plotting
from functools import reduce
from enum import Enum


###############################################################################
class Base(Enum):
    Linear = 0
    Quadratic = 1


###############################################################################
# Helper functions
def mdot(*args):
    # Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot
    return reduce(np.dot, args)


def prepend_one(X):
    # prepend a one vector to X."""
    return np.column_stack([np.ones(X.shape[0]), X])


def grid2d(start, end, num=50):
    # """reate an 2D array where each row is a 2D coordinate."""
    # np.meshgrid is pretty annoying!

    dom = np.linspace(start, end, num)
    X0, X1 = np.meshgrid(dom, dom)
    return np.column_stack([X0.flatten(), X1.flatten()])


def calcMeanSquareError(predictedY, realY):
    """calculates the square error"""
    errorSum = 0
    for i in range(len(predictedY)):
        errorSum += (predictedY[i] - realY[i]) ** 2
    return errorSum


def transformQuadraticFeature(X):
    """takes Input X and replaces it with quadratic feature"""
    dim = X.shape[1]
    transformedX = np.arange(X.shape[0] * int((1 + dim + (dim * (dim + 1) / 2))), dtype=np.float64).reshape(X.shape[0],
                                                                                                            int(
                                                                                                                1 + dim + (
                                                                                                                            dim * (
                                                                                                                                dim + 1) / 2)))

    for data_pt in range(X.shape[0]):

        transformed_x = []
        transformed_x.append(1)

        for d in range(X.shape[1]):
            transformed_x.append(X[data_pt, d])

        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                transformed_x.append(X[data_pt, i] * X[data_pt, j])

        transformedX[data_pt] = transformed_x

    return transformedX


def crossvalidation(data, subsetcount, regParam):
    # X, y = data[:, :2], data[:, 2]

    baseType = Base.Quadratic

    # print("X.shape:", X.shape)
    # print("y.shape:", y.shape)
    errors = []
    for i in range(subsetcount):
        subsets = np.vsplit(data, subsetcount)

        if (i > 0):
            tempLearn = np.vstack(subsets[0:i])

        if (i < subsetcount - 1):
            tempLearn2 = np.vstack(subsets[i + 1:])
        if (i > 0 and i < subsetcount - 1):
            learnData = np.vstack((tempLearn, tempLearn2))
        elif (i == 0):
            learndata = tempLearn2
        else:
            learndata = tempLearn
        testData = subsets[i]

        X, y = learndata[:, :2], learndata[:, 2]

        # prep for linear reg.
        if (baseType is Base.Quadratic):
            X = prepend_one(X)
        else:
            # prep for quadratic reg.
            X = transformQuadraticFeature(X)

        id_matrix = np.identity(X.shape[1])  # prepend_one later
        # ridgebeta parameter
        beta_ = mdot(inv((dot(X.T, X) + regParam * id_matrix)), X.T, y)
        expectedRes = mdot(X, beta_)
        errors.append(calcMeanSquareError(expectedRes, y))



    return (1 / subsetcount) * np.sum(errors)


###############################################################################
# load the data
data = np.loadtxt("dataQuadReg2D_noisy.txt")
print("data.shape:", data.shape)
np.savetxt("tmp.txt", data)  # save data if you want to
# split into features and labels
X, y = data[:, :2], data[:, 2]

regParam = 5
baseType = Base.Quadratic

regParam = 1
# for i in range(-4,10,):

#   print("regParam= ",regParam,"  ",crossvalidation(data,10,regParam))
#    regParam=10**i


print("X.shape:", X.shape)
print("y.shape:", y.shape)
# 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # the projection arg is important!
ax.scatter(X[:, 0], X[:, 1], y, color="red")
ax.set_title("raw data")
plt.draw()  # show, use plt.show() for blocking

# prep for linear reg.
if (baseType is Base.Linear):
    X = prepend_one(X)
else:
    # prep for quadratic reg.
    X = transformQuadraticFeature(X)

print("X.shape:", X.shape)
# Fit model/compute optimal parameters beta
# beta_ = mdot(inv(dot(X.T, X)), X.T, y)

id_matrix = np.identity(X.shape[1])  # prepend_one later
id_matrix[0, 0] = 0  # if not regularized
# ridgebeta parameter
beta_ = mdot(inv((dot(X.T, X) + regParam * id_matrix)), X.T, y)

print("Optimal beta:", beta_)
# prep for prediction
if (baseType is Base.Linear):
    X_grid = prepend_one(grid2d(-3, 3, num=30))
else:
    X_grid = transformQuadraticFeature(grid2d(-3, 3, num=30))
print("X_grid.shape:", X_grid.shape)
# Predict with trained model

y_grid = mdot(X_grid, beta_)
print("Y_grid.shape", y_grid.shape)
# vis the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # the projection part is important
ax.scatter(X_grid[:, 1], X_grid[:, 2], y_grid)  # don’t use the 1 infront

ax.scatter(X[:, 1], X[:, 2], y, color="red")  # also show the real data
ax.set_title("predicted data")
plt.show()

# calculate error
expectedRes = mdot(X, beta_)
print(calcMeanSquareError(expectedRes, y))


min=0
min_index=0
for i in range(0,1000):
    print(crossvalidation(data, 20, i/100))


