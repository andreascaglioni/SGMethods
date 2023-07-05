import numpy as np

def coordUnitVector(dim, entry):
    en = np.zeros(dim, dtype=int)
    en[entry]=1
    return en


def lexicSort(I):
    return I[np.lexsort(np.rot90(I))]

def rate(err, nDofs):
    err = np.array(err)
    nDofs = np.array(nDofs)
    return -np.log(err[1::]/err[:-1:])/np.log(nDofs[1::]/nDofs[:-1:])