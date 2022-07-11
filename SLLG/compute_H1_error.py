from dolfin import *
import numpy as np
from scipy.interpolate import interp1d


# compute reference error \norm{m_h(t)-m_reference(t)}{H1(D)} for t=n*tauref
def compute_H1_error(p, matdofsref, V3ref, ttref, matdofscurr, V3, ttcurr):
    assert matdofsref.shape[0] == V3ref.dim()
    assert matdofsref.shape[1] == len(ttref)
    assert matdofscurr.shape[0] == V3.dim()
    assert matdofscurr.shape[1] == len(ttcurr)
    assert p < 3

    # 1 interpolate current approximation on fine timegrid
    if p == 1:
        F = interp1d(ttcurr, matdofscurr, assume_sorted=True)
    elif p == 2:
        F = interp1d(ttcurr, matdofscurr, assume_sorted=True, kind='quadratic')
    else:
        error("no interpolation for time order", p)
    time_interp_mh = F(ttref)

    # 2 write current approximation in fine FE space
    err_times = np.zeros(len(ttref))
    if V3.dim() != V3ref.dim():
        M = PETScDMCollection.create_transfer_matrix(V3, V3ref)
    err = Function(V3ref)
    tmp = Function(V3)
    for niter in range(len(ttref)):
        if V3.dim() == V3ref.dim():  # here assume FE space are NESTED
            err.vector()[:] = matdofsref[:, niter] - time_interp_mh[:, niter]
        else:
            tmp.vector()[:] = time_interp_mh[:, niter]
            v2 = M * tmp.vector()
            err.vector()[:] = matdofsref[:, niter] - v2[:]
        err_times[niter] = norm(err, 'H1')
    return err_times
