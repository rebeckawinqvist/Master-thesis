import osqp
import numpy as np
import scipy as sp
from scipy import sparse
from numpy import savetxt, loadtxt
import sys
import random
from polytope import Polytope
import matplotlib.pyplot as plt

example = str(sys.argv[1]).upper()
ntrajs = int(sys.argv[2])
NT = int(sys.argv[3])

example_name = 'ex'+example
filename = example_name+'_'

# load data
A = np.loadtxt(filename+'A.csv', delimiter=',')
B = np.loadtxt(filename+'B.csv', delimiter=',')
H = np.loadtxt(filename+'cinfH.csv', delimiter=',')
xlb = np.loadtxt(filename+'xlb.csv', delimiter=',')
xub =  np.loadtxt(filename+'xub.csv', delimiter=',')
ulb = np.loadtxt(filename+'ulb.csv', delimiter=',')
uub = np.loadtxt(filename+'uub.csv', delimiter=',')

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

initial_states = np.loadtxt(filename+"initial_states_{}.csv".format(ntrajs), delimiter=',')

s = 0
for sample in initial_states:
  # Initial and reference states
  x0 = sample
  xr = np.array([0.,0.])

  if n == 4:
    print("n")
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
  prob.setup(P, q, A, l, u, warm_start=True)


  # Simulate in closed loop
  traj = [x0]
  traj_matrix = np.zeros((NT,n))
  traj_matrix[0,:] = x0
  u_matrix = np.zeros((NT-1,m))
  for i in range(NT-1):
      # Solve
      res = prob.solve()

      # Check solver status
      if res.info.status != 'solved':
          raise ValueError('OSQP did not solve the problem!')

      # Apply first control input to the plant
      ctrl = res.x[-N*nu:-(N-1)*nu]
      x0 = Ad.dot(x0) + Bd.dot(ctrl)

      # Update initial state
      l[:nx] = -x0
      u[:nx] = -x0
      prob.update(l=l, u=u)

      traj.append(x0)
      traj_matrix[i+1,:] = x0
      u_matrix[i,:] = ctrl


  # -------------------------------------------------
  # ----------------- PLOT TRAJECTORY --------------- 
  # -------------------------------------------------
  A_p = H[:,0:-1]
  b_p = H[:,-1]
  polytope = Polytope(A_p, b_p)

  if n <= 2 and traj:
    polytope.plot_poly(xlb, xub, infeasible_states = traj, show = False)

    title = "Example {}\n Sample: {} \n MPC trajectory".format(example, s+1)
    xlabel = "$x_1$"
    ylabel = "$x_2$"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 
    filen_fig = filename+"mpc_trajectory_{}_N_{}".format(s+1,NT)+".png"

    plt.savefig(filen_fig)
    #plt.show()

  np.savetxt(filename+'mpc_trajectory_{}_N_{}'.format(s+1,NT)+".csv", traj_matrix, delimiter=',')
  np.savetxt(filename+'mpc_controls_trajectory_{}_N_{}'.format(s+1,NT)+".csv", u_matrix, delimiter=',')
  s += 1

#savetxt(filenameIn, feasible_states, delimiter = ',')
#savetxt(filenameOut, feasible_inputs, delimiter = ',')