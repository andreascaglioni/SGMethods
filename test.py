import numpy as np
def lev2knots(n):
    w=np.where(n==0)
    n = n+1
    n[w]=0
    return 2**(n+1)-1

v=  np.array([0, 1,2,3,4,5])
print(lev2knots(v))