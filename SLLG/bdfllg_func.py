from dolfin import *
from scipy.special import comb
import numpy as np
import time


def bdfllg_func(r, k, alpha, T, tau, minit, VV, V3, V, W, g, m1=[], quadrature_degree=0, VERBOSE=False, Hinput=[]):
    steps = int(T / tau)

    # COEFFICIENTS BDF
    gamma = []
    delta = []
    tmp = 0
    for i in range(1, k + 1):
        tmp += 1.0 / i
    delta.append(tmp)
    for i in range(1, k + 1):
        gamma.append(comb(k, i) * (-1) ** (i - 1))
        tmp = 0
        for j in range(i, k + 1):
            tmp += comb(j, i) * (-1) ** float(i) / float(j)
        delta.append(tmp)

    # COMPUTE FIRST k TIMESTEPS
    mvec = [interpolate(minit, V3)]
    if k > 1:
        if m1:
            if VERBOSE:
                print("Iteration", k-1, ": project from reference", flush=True)
            mvec.append(m1)
        else:
            W_interp = W[0] + (W[1]-W[0]) * np.linspace(0, tau, steps**2)
            mtmp = bdfllg_func(r, k - 1, alpha, tau, tau, minit, VV, V3, V, W_interp, g, [], quadrature_degree, VERBOSE=VERBOSE, Hinput=Hinput)
            mvec.append(mtmp[-1])

    # COMPUTE REMAINING TIMESTEPS
    vlam = Function(VV)
    for j in range(k, steps + 1):
        if VERBOSE:
            print("Iteration", j, flush=True)

        # update mhat and the right-hand side part mr
        mhat = Function(V3)
        mr = Function(V3)
        for i in range(0, k):
            mhat.vector()[:] = mhat.vector()[:] + float(gamma[i]) * mvec[j - i - 1].vector()[:]
            mr.vector()[:] = mr.vector()[:] - float(delta[i + 1]) * mvec[j - (i + 1)].vector()[:]
        mr.vector()[:] = mr.vector()[:] / float(delta[0])
        mhat = mhat / sqrt(dot(mhat, mhat))

        # define variational problem
        (v, lam) = TrialFunctions(VV)
        (phi, mu) = TestFunctions(VV)

        # avoid just-in-time compilation at every timestep
        Cs = Constant(float(np.sin(W[j])))
        Cc = Constant(float(1 - np.cos(W[j])))

        # build external field
        if Hinput:
            H = Constant(Hinput)
        else:
            H = Constant((0., 0., 1.))
        HH = -Cs * cross(H, g) + Cc * cross(cross(H, g), g)  # original: float(np.sin(-W[j])) * cross(H, g) + float(1 - np.cos(-W[j])) * cross(cross(H, g), g)

        # define LLG form
        if quadrature_degree == 0:
            dxr = dx
        else:
            dxr = dx(metadata={'quadrature_degree': quadrature_degree})
        if VERBOSE:
            print("Assembling...", flush=True)
        beg = time.time()
        lhs = ((alpha * inner(v, phi)
                + inner(cross(mhat, v), phi)
                + Constant(tau / float(delta[0])) * inner(nabla_grad(v + Cs * cross(v, g) + Cc * cross(cross(v, g), g)),
                                                          nabla_grad(phi + Cs * cross(phi, g) + Cc * cross(cross(phi, g), g)))) * dxr
               + inner(dot(phi, mhat), lam) * dxr
               + inner(dot(v, mhat), mu) * dxr)
        rhs = (-inner(
            nabla_grad(mr + Cs * cross(mr, g) + Cc * cross(cross(mr, g), g)),
            nabla_grad(phi + Cs * cross(phi, g) + Cc * cross(cross(phi, g), g)))
               + inner(H + HH, phi)) * dxr

        # assembly
        A = assemble(lhs)
        b = assemble(rhs)
        end = time.time()
        if VERBOSE:
            print(end - beg, "s")

        # compute solution
        if VERBOSE:
            print("Solving linear system...", flush=True)
        beg = time.time()
        solver = PETScKrylovSolver()
        solver.set_from_options()
        solver.solve(A, vlam.vector(), b)
        # solve(A, vlam.vector(), b, 'gmres', 'ilu')  # if you dont use ''assemble''
        end = time.time()
        if VERBOSE:
            print(end - beg, "s")

        # update magnetization
        (v, lam) = vlam.split(deepcopy=True)
        mvec.append(Function(V3))
        mvec[j].vector()[:] = mr.vector()[:] + tau / float(delta[0]) * v.vector()[:]
    return mvec


def bdfllg_func_original(r, k, N, alpha, T, tau, minit, mesh, VV, V3, V, W):
    steps = int(T / tau)

    # precomputation for the BDF scheme
    gamma = []
    delta = []

    tmp = 0
    print(k)
    for i in range(1, k + 1):
        tmp += 1.0 / i;
    delta.append(tmp)

    for i in range(1, k + 1):
        gamma.append(comb(k, i) * (-1) ** (i - 1))

        tmp = 0
        for j in range(i, k + 1):
            tmp += comb(j, i) * (-1) ** float(i) / float(j)
        delta.append(tmp)

    print(delta)
    print(gamma)

    # create list of previous time steps
    mvec = []
    mvec.append(interpolate(minit, V3))
    if k > 1:
        # mvec_tmp=bdfllg_func(r,k-1,N,alpha,(k-1)*tau,tau**(float(k)/(k-1)),minit,mesh,VV,V3,V,[W[0],W[1]]);
        mvec_tmp = bdfllg_func_original(r, k - 1, 1, alpha, tau, tau, minit, mesh, VV, V3, V, [W[0], W[1]]);

    for i in range(1, k):
        # mvec.append(mvec_tmp[int(i*tau**(float(-1)/(k-1)))])

        mvec.append(mvec_tmp[1])

    # Time stepping\arabic{}
    for j in range(k, steps + 1):

        print(j)
        # update mhat and the right-hand side part mr

        mhat = interpolate(Expression(['0', '0', '0'], degree=r), V3)
        mr = interpolate(Expression(['0', '0', '0'], degree=r), V3)
        for i in range(0, k):
            mhat.vector()[:] = mhat.vector()[:] + float(gamma[i]) * mvec[j - i - 1].vector()[:]
            mr.vector()[:] = mr.vector()[:] - float(delta[i + 1]) * mvec[j - (i + 1)].vector()[:]

        mr.vector()[:] = mr.vector()[:] / float(delta[0])
        mhat = mhat / sqrt(dot(mhat, mhat))

        # for i in range(0,k):
        #    mhat=mhat + float(gamma[i])*mvec[j-i-1]
        #    mr=mr - float(delta[i+1])*mvec[j-(i+1)]
        #
        # mr=mr/float(delta[0])
        # mr=project(mr,V3)
        # mhat=mhat/sqrt(dot(mhat,mhat))

        # define variational problem
        (v, lam) = TrialFunctions(VV)
        (phi, mu) = TestFunctions(VV)

        # build external field
        H = Constant((0, 1, 1))  # Constant((0,0,0.00000000001)) #
        g = Constant((2 / 3, 1 / 3, 2 / 3))
        HH = float(np.sin(-W[j])) * cross(H, g) + float(1 - np.cos(-W[j])) * cross(cross(H, g), g)

        # reducing quadrature degree
        # dx = dx(metadata={'quadrature_degree': 5})

        # define LLG form
        dxr = dx(metadata={'quadrature_degree': 5})
        lhs = ((alpha * inner(v, phi) + inner(cross(mhat, v), phi)
                + tau / float(delta[0]) * inner(
                    nabla_grad(v + float(np.sin(W[j])) * cross(v, g) + float(1 - np.cos(W[j])) * cross(cross(v, g), g)),
                    nabla_grad(
                        phi + float(np.sin(W[j])) * cross(phi, g) + float(1 - np.cos(W[j])) * cross(cross(phi, g),
                                                                                                    g)))) * dxr
               + inner(dot(phi, mhat), lam) * dxr + inner(dot(v, mhat), mu) * dxr)

        rhs = (-inner(
            nabla_grad(mr + float(np.sin(W[j])) * cross(mr, g) + float(1 - np.cos(W[j])) * cross(cross(mr, g), g)),
            nabla_grad(
                phi + float(np.sin(W[j])) * cross(phi, g) + float(1 - np.cos(W[j])) * cross(cross(phi, g), g))) + inner(
            H + HH, phi)) * dxr

        # compute solution
        vlam = Function(VV)
        solve(lhs == rhs, vlam, solver_parameters={"linear_solver": "gmres"},
              form_compiler_parameters={"optimize": True})

        # update magnetization
        (v, lam) = vlam.split(deepcopy=True)
        mvec.append(interpolate(Expression(['0', '0', '0'], degree=r),
                                V3));  # project(mj, V3, solver_type='cg',preconditioner_type="ilu"));
        mvec[j].vector()[:] = mr.vector()[:] + tau / float(delta[0]) * v.vector()[:];

        # plot current solution
        # dolfin.plot(m)
        # plt.show()

    return mvec;
