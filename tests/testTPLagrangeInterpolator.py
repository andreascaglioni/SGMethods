import sys
sys.path.append('/home/andrea/workspace/SGMethods')
from SGMethods.TPLagrangeInterpoator import TPLagrangeInterpolator
from SGMethods.ScalarNodes import unboundedKnotsNested
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# choose function
N = 2
dimF = 3
f = lambda x, y: np.sin(x+y)*np.array([1.,2.,3.])

NRNDSamples = 1000
xxRND = np.random.normal(0, 1, [NRNDSamples, N])
ww = np.exp(-0.5*np.linalg.norm(xxRND, axis=1)**2)
spaceNorm = lambda x: np.linalg.norm(x, ord=2, axis=1)
fOnxx = np.zeros((NRNDSamples, dimF))
for w in range(NRNDSamples):
    fOnxx[w] = f(xxRND[w, 0], xxRND[w, 1])


xx = unboundedKnotsNested(3)
nodesTuple = (xx, xx)
xg, yg = np.meshgrid(xx, xx, indexing='ij', sparse=True)
F = f(xg, yg)

I = TPLagrangeInterpolator(nodesTuple, F)
yy = I(xxRND)

# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xxRND[:,0], xxRND[:,1], yy)
# ax.scatter(xxRND[:,0], xxRND[:,1], fOnxx)
# plt.show()
print(np.amax(spaceNorm(yy - fOnxx)*ww))