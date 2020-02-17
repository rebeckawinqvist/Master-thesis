import osqp
import numpy as np
import scipy as sp
from scipy import sparse
from numpy import savetxt, loadtxt

# Discrete time model of a quadcopter
eps = 0.1
Ad = sparse.csc_matrix([
  [1.,      1. ],
  [0.,      1.  ]
])

Bd = sparse.csc_matrix([
  [0. ],
  [1. ]
])

[nx, nu] = Bd.shape

# Constraints
u0 = 0
umin = np.array([-0.5]) - u0
umax = np.array([0.5]) - u0
xmin = np.array([-5.,
                 -5.])
xmax = np.array([ 5.,
                 5.])



# Objective function
Q = sparse.diags([1., 1.])
QN = Q
R = 10

# Initial and reference states
x0 = np.array([-4.5, 2.])
xr = np.array([0.,0.])

# Prediction horizon
N = 3

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
P_flat = P.toarray().flatten()
A_flat = A.toarray().flatten()
q_flat = q.flatten()
l_flat = l.flatten()
u_flat = u.flatten()
#size = P_flat.shape[0] + A_flat.shape[0] + q_flat.shape[0] + l_flat.shape[0] + u_flat.shape[0] + x0.shape[0]
size = x0.shape[0]

data_generation = 'grid'
filenameIn = ''
filenameOut = ''

if data_generation == 'grid':
  initial_states = np.loadtxt('initial_states_grid.csv', delimiter=',')
  filenameIn = 'input_data_grid.csv'
  filenameOut = 'output_data_grid.csv'

elif data_generation == 'rays':
  initial_states = np.loadtxt('initial_states_rays.csv', delimiter=',')
  filenameIn = 'input_data_rays.csv'
  filenameOut = 'output_data_rays.csv'


nsim = initial_states.shape[0]
feed_data = np.zeros((nsim, size))
result_data = np.zeros((nsim, N*nu))

i = 0
for row in initial_states:
  # Solve
  x0 = row

  leq = np.hstack([-x0, np.zeros(N*nx)])
  ueq = leq
  l = np.hstack([leq, lineq])
  u = np.hstack([ueq, uineq])
  prob = osqp.OSQP()

  prob.setup(P, q, A, l, u)

  res = prob.solve()
  u_res = res.x[-N*nu:]

  # Check solver status
  if res.info.status != 'solved':
    raise ValueError('OSQP did not solve the problem!')
  
  # Save data
  in_data = x0
  feed_data[i,:] = in_data
  result_data[i,:] = u_res

  # update initial state

  i += 1
  if i == nsim:
    break


savetxt(filenameIn, feed_data, delimiter = ',')
savetxt(filenameOut, result_data, delimiter = ',')