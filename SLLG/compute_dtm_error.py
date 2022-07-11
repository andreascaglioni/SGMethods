from dolfin import *
import numpy as np
from scipy.special import comb
from scipy.interpolate import interp1d


# compute reference error \sum_t \norm{\partial_t (m_reference(t) - m_h(t))}{L2(D)} for t=n*tauref
def compute_dtm_error(k, urefdofsmat, V3ref, tauref, ttref, uhdofsmat, V3, tau, tt):
    assert urefdofsmat.shape[0] == V3ref.dim()
    assert urefdofsmat.shape[1] == len(ttref)
    assert uhdofsmat.shape[0] == V3.dim()
    assert uhdofsmat.shape[1] == len(tt)
    assert k < 3
    assert urefdofsmat.shape[1] >= 2*k

    # 1  coefficients BDF order k
    delta = []
    tmp = 0
    for i in range(1, k + 1):
        tmp += 1.0 / i
    delta.append(tmp)
    for i in range(1, k + 1):
        tmp = 0
        for j in range(i, k + 1):
            tmp += comb(j, i) * (-1) ** float(i) / float(j)
        delta.append(tmp)

    # 2 compute dt u_ref
    dturef = []
    for niter in range(len(delta)):
        dturef.append(Function(V3ref))
        for j in range(len(delta)):
            dturef[niter].vector()[:] -= delta[j] * urefdofsmat[:, niter+j]  # flipped points, must change sign
        dturef[niter].vector()[:] *= 1/tauref
    for niter in range(len(delta), urefdofsmat.shape[1]):
        dturef.append(Function(V3ref))
        for j in range(len(delta)):
            dturef[niter].vector()[:] += delta[j] * urefdofsmat[:, niter-j]
        dturef[niter].vector()[:] *= 1/tauref

    # 3 compute dt uh
    dtuhmat = np.zeros((V3.dim(), len(tt)))
    for niter in range(len(delta)):
        for j in range(len(delta)):
            dtuhmat[:, niter] -= delta[j] * uhdofsmat[:, niter+j]  # flipped points, must change sign
    for niter in range(len(delta), uhdofsmat.shape[1]):
        for j in range(len(delta)):
            dtuhmat[:, niter] += delta[j] * uhdofsmat[:, niter-j]
    dtuhmat *= 1/tau

    # 4 interpolate dt uh (piecewise const for k=1, affine for k=2) on reference timegrid
    if k == 1:
        F = interp1d(tt, dtuhmat, assume_sorted=True, kind='previous')
    elif k == 2:
        F = interp1d(tt, dtuhmat, assume_sorted=True, kind='linear')
    else:
        error('not implemented k>2')
    dtuh_reft_mat = F(ttref)

    # 5 express it in finer FE space and subtract from dt m_ref
    if V3.dim() != V3ref.dim():
        M = PETScDMCollection.create_transfer_matrix(V3, V3ref)
        tmp = Function(V3)
    for niter in range(len(dturef)):
        if V3.dim() == V3ref.dim():  # here assume FE space are NESTED
            dturef[niter].vector()[:] -= dtuh_reft_mat[:, niter]
        else:
            tmp.vector()[:] = dtuh_reft_mat[:, niter]
            v = M * tmp.vector()
            dturef[niter].vector()[:] -= v[:]

    # compute L2 error
    err_times = np.zeros(len(ttref))
    for niter in range(len(err_times)):
        err_times[niter] = norm(dturef[niter], 'L2')
    return err_times
