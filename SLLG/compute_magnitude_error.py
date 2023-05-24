from dolfin import *
import numpy as np


# compute error \vert 1 - \vert m_h(t) \vert\vert \forall t = n*tau
def compute_magnitude_error(Ntau, mapr, V):
    errmagtime = []
    for niter in range(Ntau + 1):
        v = mapr[niter]
        v.vector()[:] *= v.vector()
        (a, b, c) = v.split(deepcopy=True)
        a.vector().axpy(1, b.vector())
        a.vector().axpy(1, c.vector())
        a.vector().set_local(np.sqrt(a.vector().get_local()))
        errfun = interpolate(Expression('1', degree=1), V)
        errfun.vector().axpy(-1, a.vector())
        errmagtime.append(norm(errfun, 'L2'))
    return errmagtime
