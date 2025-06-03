"""Generate a cool figure for my webpage and GitHub README.md"""

from math import pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys
from os.path import join

path_sgmethods = "../sgmethods"
sys.path.append(path_sgmethods)
from sgmethods.sparse_grid_interpolant import SGInterpolant
from sgmethods.multi_index_sets import aniso_smolyak_mid_set
from sgmethods.nodes_1d import cc_nodes
from sgmethods.utils import plot_2d_midset


# DATA
C_f = 4
domain = [[-1, 1], [-1, 1]]
def f(x):
    # return np.exp(x[0])*np.exp(C_f*x[1])  
    return np.sin(0.5*pi*(x[1]+0.5)) * np.sin((C_f)*pi*x[0])

# INTERPOLATE
a_aniso = [0.72, 1]
w = 3.6
midset = aniso_smolyak_mid_set(w, N=2, a=a_aniso)
I = SGInterpolant(midset, cc_nodes, lambda x: 2**x+1, verbose=True)
f_sg = I.sample_on_SG(f)
xx = np.random.random((50, 2))*2-1
If_xx = I.interpolate(xx, f_sg)

# ERROR
err = np.amax(np.abs(f(xx.T) - If_xx))
print("#SG:", I.num_nodes, "Error:", err)

# PLOT
fig = plt.figure("midset")
plot_2d_midset(midset)

fig = plt.figure("f")
ax= fig.add_subplot()
x = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, x)
F = f([X, Y])
p = ax.pcolor(X, Y, F, cmap='viridis', vmin=(F).min(), vmax=(F).max())
cb = fig.colorbar(p, ax=ax)

ax.plot(I.SG[:, 0], I.SG[:, 1], '.', color='red', markersize=15)

ax.set_xlabel('x')
ax.set_ylabel('y')
plt.tight_layout()
plt.show()