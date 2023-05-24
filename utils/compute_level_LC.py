import numpy as np
from math import floor, log2

'''Compte level and index in LC expansion given linear index
INPUT i int number of basis function
OUTPUT l int level of i-th basis function NB starts from 0
       j int index of i-th basis function within l-th level NB starts from 1'''

def compute_level_LC(i):
    if i == 0:
        l=0
        j=1
    elif i == 1:
        l=1
        j=1
    else:
        l = floor(log2(i))+1
        j = i - 2**(l-1)
    return l, j