import matplotlib as mpl
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    return errorSum / len(predictedY)


def transformQuadraticFeature(X):
    """takes Input X and replaces it with quadratic feature"""
    dim = X.shape[1]
    transformedX = np.arange(X.shape[0] * int((dim + (dim * (dim + 1) / 2))), dtype=np.float64).reshape(X.shape[0], int(
        dim + (dim * (dim + 1) / 2)))

    for data_pt in range(X.shape[0]):

        transformed_x = []
        # transformed_x.append(1)

        for d in range(X.shape[1]):
            transformed_x.append(X[data_pt, d])

        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                transformed_x.append(X[data_pt, i] * X[data_pt, j])

        transformedX[data_pt] = np.array(transformed_x)

    return transformedX


def calc_deltaL(beta_, plist, y):
    id_matrix = np.identity(X.shape[1])
    return dot(X.T, plist - y) + 2 * lambda_ * dot(id_matrix, beta_)


def calc_delta2L(beta_, plist, y):
    id_matrix = np.identity(X.shape[1])
    W = []
    for i in range(X.shape[0]):
        W.append(plist[i] * (1 - plist[i]))
    W = np.diag(W)
    return mdot(X.T, W, X) + 2 * lambda_ * id_matrix


def calc_plist(X, beta_):
    plist = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x = X[i].T
        plist[i] = sigmoid(dot(x, beta_))
    return plist


# load the data
data = np.loadtxt("data2Class.txt")
print("data.shape:", data.shape)
np.savetxt("tmp.txt", data)  # save data if you want to
# split into features and labels
X, y = data[:, :2], data[:, 2]

colors = ['red', 'blue']
levels = [0, 1]
cmap, norm = mpl.colors.from_levels_and_colors(levels=levels, colors=colors, extend='max')

###########################################
# Input parameters
lambda_ = 1
baseType = Base.Quadratic
###################################################################
if (baseType is Base.Quadratic):
    X = transformQuadraticFeature(X)

beta_ = np.zeros(X.shape[1])
print(X.shape[1])
for i in range(20):
    plist = calc_plist(X, beta_)
    deltaL = calc_deltaL(beta_, plist, y)
    delta2L = calc_delta2L(beta_, plist, y)
    beta_ = beta_ - dot(inv(delta2L), deltaL)

print(beta_)
print("X.shape:", X.shape)
print("y.shape:", y.shape)

if (baseType is Base.Linear):
    X_grid = grid2d(-3, 3, num=50)
else:
    X_grid = transformQuadraticFeature(grid2d(-3, 3, num=50))

y_grid = mdot(X_grid, beta_)

sigmovec = np.vectorize(sigmoid)

plist = sigmovec(y_grid)

# 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111)  # the projection arg is important!

ax.scatter(X_grid[:, 0], X_grid[:, 1], c=plist, cmap=plt.cm.get_cmap('RdBu'), alpha=0.5)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, norm=norm)  # also show the real data
ax.set_title("2d p(x) plot")

plt.draw()  # show, use plt.show() for blocking
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # the projection part is important
ax.scatter(X_grid[:, 0], X_grid[:, 1], plist, color='green')  # don’t use the 1 infront
ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap=cmap, norm=norm)  # also show the real data
ax.set_title("3d p(x) plot")
plt.show()
