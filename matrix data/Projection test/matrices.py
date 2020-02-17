import osqp
import numpy as np
import scipy as sp
from scipy import sparse
from numpy import savetxt, loadtxt


A = np.loadtxt('A.csv', delimiter=',')
B = np.loadtxt('B.csv', delimiter=',')
H = np.loadtxt('cinfH.csv', delimiter=',')

u_low = -0.5
u_high = 0.5
C_c = H[:,0:2]
d_c = H[:,2]
C_u = np.array([1., -1.])
d_u = np.array([u_high, -u_low])

E = C_c @ B

print("A: \n", A)
print("\n B \n: ", B)
print("\n H: \n", H)
print("\n C_c: \n", C_c)
print("\n E: \n", E)
print("\n C_u: \n", C_u)
print("\n d_c: \n", d_c)
print("\n d_u: \n", d_u)