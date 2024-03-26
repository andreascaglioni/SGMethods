from __future__ import division
import numpy as np
from math import sqrt, log2, pi, factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from multiprocessing import Pool
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.ScalarNodes import unboundedKnotsNested, equispacedNodes, CCNodes, HermiteNodes
from SGMethods.MidSets import midSet, TPMidSet, SmolyakMidSet, anisoSmolyakMidSet, hyperbolicCrossMidSet
from SGMethods.SGInterpolant import SGInterpolant



"""Plot a multi-index set and corresponding sparse grid given nodes, midset, lev2knot map"""


#  TP interpolantion w equispaced nodes #############################################
# lev2knots = lambda n: 2**(n)+1
# knots = lambda m : equispacedNodes(m)
# midset = TPMidSet(3, N=2)


#  SG interpolantion w equispaced nodes #############################################
# lev2knots = lambda n: 2**(n)+1
# knots = lambda m : equispacedNodes(m)
# midset = SmolyakMidSet(4, N=2)

#  SG interpolantion w CC nodes #############################################
# lev2knots = lambda n: 2**(n)+1
# knots = lambda m : CCNodes(m)
# midset = SmolyakMidSet(4, N=2)

#  aniso SG interpolantion w CC nodes #############################################
# lev2knots = lambda n: 2**(n)+1
# knots = lambda m : CCNodes(m)
# midset = anisoSmolyakMidSet(2, N=2, a=np.array([0.5, 1]))


# Hermite nodes with a random anisotropic midset
lev2knots = lambda n:n+1
knots = lambda m : HermiteNodes(m)
midset = anisoSmolyakMidSet(4, N=2, a=np.array([0.5, 1]))



#  Hyperbolc cross #############################################
# lev2knots = lambda n: 2**(n)+1
# knots = lambda m : equispacedNodes(m)
# midset = hyperbolicCrossMidSet(5, N=2)

#  pwPoly on unbounded domain ####################################################################
# lev2knots = lambda n: 2**(n+1)-1
# knots = lambda m : unboundedKnotsNested(m, p=3)
# midset = np.array([[0,0],
#                    [0,1],
#                    [0,2],
#                    [1,0],
#                    [1,1],
#                    [2,0],
#                    [3,0]
#                    ])

#  define interpolant, get nodes ###################################################################
interpolant = SGInterpolant(midset, knots, lev2knots, interpolationType="linear", NParallel=1)
xx = interpolant.SG

# plot SG ##########################################################################################
plt.plot(xx[:,0], xx[:,1] ,'.', markersize=15)
# plt.axhline(0, -5, 5, color="black")
# plt.axvline(0, -5, 5, color="black")
# plt.title("Sparse grid "+r'$\mathcal{Y}(\Lambda)$', fontsize=25)
# ax = plt.gca()
# ax.axis('off')
plt.show()

# plot Midset
plt.plot(midset[:,0], midset[:,1], 's', markersize=15)
# plt.title("Multi-index set "+r'$\Lambda$', fontsize=35)
# for i in range(4):
#     plt.text(i-0.05,-0.25 , i, fontsize=20)
# for i in range(3):
#     plt.text(-0.25, i-0.05, i, fontsize=20)
# ax.axis('off')
# ax = plt.gca()
# ax.axis('off')
# plt.axhline(0, 0, 4, color="black")
# plt.axvline(0, 0, 3, color="black")
plt.show()





# EXPORT ########################################################################################
np.savetxt("TDanisoHermite_midset.csv", midset,delimiter=",")
np.savetxt("TDanisoHermite_SG.csv", xx,delimiter=",")