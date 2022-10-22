from SGMethods.ScalarNodes import unboundedNodesOptimal
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def rate(nNodes, err):
    err = np.array(err)
    nNodes = np.array(nNodes)
    return -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])


p=1
lev2knot = lambda nu : 2**(nu+1)-1
M=[]
nNodes=[]
for nu in range(10):
    nNodesCurr = lev2knot(nu+1)
    halfNodes = int(0.5*(nNodesCurr-1))
    yy = unboundedNodesOptimal(nNodesCurr, p)
    vv = np.sqrt(norm.pdf(yy[halfNodes:-1:])) * (yy[halfNodes+1:]-yy[halfNodes:-1])**(2*p)
    M.append(max(vv))
    nNodes.append(nNodesCurr)
print(rate(nNodes, M))