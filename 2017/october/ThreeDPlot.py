import mpl_toolkits.mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math

x, y = np.mgrid[0:1:50j, 0:1:50j]
# z = x * np.exp(-x ** 2 - y ** 2)

z = np.sqrt(1 - pow(x, 2) - pow(y, 2))
ax = plt.subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=2, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('p')

plt.show()
