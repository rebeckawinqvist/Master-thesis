import osqp
import numpy as np
import scipy as sp
from scipy import sparse
from numpy import savetxt, loadtxt
import sys
import random
from polytope import Polytope
from datetime import datetime

example = str(sys.argv[1]).upper()
gen = str(sys.argv[2])
nsamples = int(sys.argv[3])

if len(sys.argv) > 4:
    date = sys.argv[4]
else:
    date = datetime.date(datetime.now())


gen_type = ''
if gen.upper() == "TRAIN" or gen.upper() == "TEST":
    gen_type = gen.lower()
else:
    raise ValueError

example_name = 'ex'+example
folder_name = 'ex'+example+'/'
filename = folder_name+example_name+'_'

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

# Show problem setup
print("Running example: {}".format(example.upper()))
print("Q: \n", Q, "\n")
print("QN: \n", QN, "\n")
print("R: \n", R, "\n")
print("N: \n", N, "\n")


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


initial_states = np.loadtxt("ex{}/initial_states/ex{}_initial_states_{}_{}.csv".format(example, example, nsamples, gen_type), delimiter=',')


# Initial and reference states
x0 = np.array([-4.5, 2.])
xr = np.array([0.,0.])

if n == 4:
  print("4D example")
  rows = initial_states.shape[0]
  ridx = random.randint(0, rows-1)
  x0 = initial_states[ridx,:]
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
prob.setup(P, q, A, l, u, warm_start=True, verbose=False)

# Simulate in closed loop
size = x0.shape[0]


nsim = initial_states.shape[0]
states = np.zeros((nsim, size))
control_inputs = np.zeros((nsim, N*nu))

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
  states[i,:] = in_data
  control_inputs[i,:] = u_res

  # update initial state

  i += 1
  if i == nsim:
    break



# -------------------------------------------------
# ----------------- VERIFY DATA ------------------- 
# -------------------------------------------------
first_input = control_inputs[:,0:m]

A_p = H[:,0:-1]
b_p = H[:,-1]
polytope = Polytope(A_p, b_p)

feasible_states = []
feasible_inputs = []
infeasible_states = []
infeasible_transitions = []

for (x, u_N) in zip(states, control_inputs):
  # check current state
  x = np.array(x)
  u = np.array(u_N[0:m])
  if not polytope.is_inside(x):
    print("State not inside: ", x, "\n")

  else:
    # check next state
    if m == 1:
      x1 = A_sys @ x + B_sys * u
    else:
      x1 = A_sys @ x + B_sys @ u
    if not polytope.is_inside(x1):
      #print("Next state not inside. \nx:\n {}, \nx1:\n {}".format(x, x1))
      infeasible_states.append(x1)
      infeasible_transitions.append((x,u))
    else:
      feasible_states.append(x)
      feasible_inputs.append(u_N)

if n <= 2:
  polytope.plot_poly(xlb, xub, infeasible_states = infeasible_states)


print("Infeasible data [%]: ", 100*len(infeasible_states)/len(states))
print("Kept data: ", len(feasible_states), "/", len(states))


filenameIn = 'ex{}/{}/input_data/ex{}_input_data_nsamples_{}_{}.csv'.format(example, date, example, nsamples, gen_type)
filenameOut = 'ex{}/{}/output_data/ex{}_output_data_nsamples_{}_{}.csv'.format(example, date, example, nsamples, gen_type)

savetxt(filenameIn, feasible_states, delimiter = ',')
savetxt(filenameOut, feasible_inputs, delimiter = ',')