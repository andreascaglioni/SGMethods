import numpy as np
import matplotlib.pyplot as plt
from math import pi 
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.MidSets import midSet
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.MidSets import anisoSmolyakMidSet

def CCNodes(npt):
    n = np.cos(np.linspace(0, npt, npt)*pi/npt )
    return n  # (n+1)/2


def plot_nodes(nodes):
    plt.plot(nodes[:,0], nodes[:,1], '.', markersize=20, c='white')
    # plt.xlim([0,1])
    # plt.ylim([0,1])
    ax = plt.gca()
    ax.set_aspect(1.)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
npt = 50

####################### MC
xMC = np.random.uniform(0,1,(npt,2))
plt.figure()
plot_nodes(xMC)
plt.savefig("mc.png",bbox_inches='tight', transparent=True)

plt.draw()
plt.pause(0.001)


# ####################### QMC lattice rule
g = np.array([1, 12])
g = g/npt
xQMCRaw = np.reshape(np.linspace(1,npt,npt), (npt, 1)) * g
xQMC = np.mod(xQMCRaw, np.ones_like(xQMCRaw))
plt.figure()
plot_nodes(xQMC)
plt.savefig("qmc.png",bbox_inches='tight', transparent=True)

plt.draw()
plt.pause(0.001)

# ####################### TP Clenshw-Curtis
npt = 7
xCC1d = CCNodes(npt)
mg = np.meshgrid(xCC1d, xCC1d)
k0 = mg[0].flatten()
k1 = mg[1].flatten()
xCCTP = np.vstack((k0, k1))
plt.figure()
plot_nodes(xCCTP.T)
plt.savefig("tpcc.png",bbox_inches='tight', transparent=True)

plt.draw()
plt.pause(0.001)

####################### Sparse grids CC
knots = lambda n : CCNodes(n)
lev2knots = lambda nu : 2**(nu+1)-1
I = anisoSmolyakMidSet(3.,2, np.array([1, 1.]))
I=np.array(I)
interpolant = SGInterpolant(I, knots, lev2knots, interpolationType='polynomial', NParallel=1)
print(interpolant.SG.shape)
plt.figure()
plot_nodes(interpolant.SG)
plt.savefig("sg.png",bbox_inches='tight', transparent=True)
plt.show()
