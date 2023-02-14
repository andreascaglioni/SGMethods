import numpy as np

def coordUnitVector(dim, entry):
    en = np.zeros(dim)
    en[entry]=1
    return en


def lexicSort(I):
    return I[np.lexsort(np.rot90(I))]