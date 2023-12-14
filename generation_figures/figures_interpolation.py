import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.ScalarNodes import unboundedNodesOptimal


"""Generate figure of 1D interpolation. Nodes, samples and a degree 2 piecewise polynomail interpolation"""

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }

font2 = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 28,
        }
# xxp = unboundedNodesOptimal(15, p=3)
# xxp =xxp[1:-1]
xxp = np.array([-4, -3, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 3, 4])

xx = xxp[::2]
y = np.array([0.7, 0.6, 0.5, 0.6, 0.8, 0.5, 0.3,  0.4, 0.7,  0.4, 0.1,  0.35,  0.6])
f = interp1d(xxp, y, kind="quadratic", bounds_error=False, fill_value="extrapolate")
xnew = np.arange(xx[0]-2, xx[-1]+2, 0.01)
 

plt.plot(xnew, f(xnew), '-', color="red")
plt.plot(xx, np.zeros_like(xx), '.', markersize=20, color="black")
plt.plot(xx, y[::2], '.', markersize=20, color="black")

plt.text(-3,  0.2 , r'$I_7^2[u]$', fontdict=font2)
for i in range(7):
    plt.text(xx[i]-0.1, -0.12 , r'$y_'+str(i+1)+'$', fontdict=font)

ax = plt.gca()
plt.axhline(0, -5, 5, color="black")
ax.axis('off')
plt.show()