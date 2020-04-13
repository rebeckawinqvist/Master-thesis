import osqp
import numpy as np
import scipy as sp
from scipy import sparse
from numpy import savetxt, loadtxt
import sys
import random
from polytope import Polytope
import matplotlib.pyplot as plt
from datetime import datetime
from makedirs import *

#example = str(sys.argv[1]).upper()
ntrajs = int(sys.argv[1])
nsim = int(sys.argv[2])
gen_type = str(sys.argv[3])
if len(sys.argv) > 4:
    date = sys.argv[4]
else:
    date = datetime.date(datetime.now())

if gen_type.upper() == "TRAIN" or gen_type.upper() == "TEST":
    gen_type = gen_type.lower()
else:
    raise ValueError


makedirs(date)


fn_save = "{}/trajectories/{}/".format(date, gen_type)

examples = ["A"]

for example in examples:
    # load 
    filename = "ex{}/ex{}_".format(example, example)

    A = np.loadtxt(filename+'A.csv', delimiter=',')
    B = np.loadtxt(filename+'B.csv', delimiter=',')
    H = np.loadtxt(filename+'cinfH.csv', delimiter=',')
    xlb = np.loadtxt(filename+'xlb.csv', delimiter=',')
    xub =  np.loadtxt(filename+'xub.csv', delimiter=',')
    ulb = np.loadtxt(filename+'ulb.csv', delimiter=',')
    uub = np.loadtxt(filename+'uub.csv', delimiter=',')

    A_p = H[:,0:-1]
    b_p = H[:,-1]
    polytope = Polytope(A_p, b_p)

    Q = np.loadtxt(filename+'Q.csv', delimiter=',')
    QN = Q
    R = np.loadtxt(filename+'R.csv', delimiter=',')
    N = int(np.loadtxt(filename+'N.csv', delimiter=','))


    n = A.shape[1]
    m = B.ndim

    xmin, xmax = xlb, xub
    umin, umax = ulb, uub
    Ad, Bd = A,B
    nx, nu = n, m


    # system dynamics
    A_sys, B_sys = A, B

    if m == 1:
        Bd = np.expand_dims(B, axis=1)

    initial_states = np.loadtxt("ex{}/initial_states/ex{}_initial_states_{}_{}.csv".format(example, example, ntrajs, gen_type), delimiter=',')

    s = 0
    feasible_states = []
    not_solved = []
    for sample in initial_states:
        # Initial and reference states
        x0 = sample
        xr = np.array([0.,0.])
        solved = True

        if n == 4:
            xr = np.array([0., 0., 0., 0.])


        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                                sparse.kron(sparse.eye(N), R)], format='csc')
        # - linear objective
        q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                        np.zeros(N*nu)])
        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])
        ueq = leq
        # - input and state constraints
        Aineq = sparse.eye((N+1)*nx + N*nu)
        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True, max_iter = 4000, verbose=False)


        # Simulate in closed loop
        traj_matrix = np.zeros((nsim,n+m))
        for i in range(nsim):
            # Solve
            res = prob.solve()

            # Check solver status
            if res.info.status != 'solved':
                raise ValueError('OSQP did not solve the problem!')

            # get first input
            ctrl = res.x[-N*nu:-(N-1)*nu]

            # store data
            traj_matrix[i,0:n] = x0
            traj_matrix[i,n:] = ctrl

            # Apply first control input to the plant
            x0 = Ad.dot(x0) + Bd.dot(ctrl)

            # Update initial state
            l[:nx] = -x0
            u[:nx] = -x0
            prob.update(l=l, u=u)

        
        np.savetxt(fn_save+"ex{}_trajectory_{}_nsim_{}_{}.csv".format(example, s+1, nsim, gen_type), traj_matrix, delimiter=',')
        s += 1


