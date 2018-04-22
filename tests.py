import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D # 3D plotting
from functools import reduce
from enum import Enum


testfeld=np.arange(25).reshape(5,5)
testfeld=np.vsplit(testfeld,5)
tester=[]

print (tester)


test=[4,3,2,1]
print(np.sum(test))



print(tester)


import numpy as np
import matplotlib.pyplot as plt


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
