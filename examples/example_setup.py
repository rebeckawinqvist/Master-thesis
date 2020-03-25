import numpy as np  
import logging
logging.getLogger().setLevel(logging.INFO)
import statistics           
import sys
import matplotlib.pyplot as plt

example = str(sys.argv[1]).upper()

filename = 'ex'+example+'/ex'+example+'_'

A = np.loadtxt(filename+'A.csv', delimiter=',')
B = np.loadtxt(filename+'B.csv', delimiter=',')
H = np.loadtxt(filename+'cinfH.csv', delimiter=',')
L = np.loadtxt(filename+'L.csv', delimiter=',')
xlb = np.loadtxt(filename+'xlb.csv', delimiter=',')
xub =  np.loadtxt(filename+'xub.csv', delimiter=',')
ulb = np.loadtxt(filename+'ulb.csv', delimiter=',')
uub = np.loadtxt(filename+'uub.csv', delimiter=',')
Q = np.loadtxt(filename+'Q.csv', delimiter=',')
QN = Q
R = np.loadtxt(filename+'R.csv', delimiter=',')
N = int(np.loadtxt(filename+'N.csv', delimiter=','))


print("A: \n", A)
print("\nB: \n", B)
print("\nx_min: \n", xlb)
print("\nx_max: \n", xub)
print("\nu_min: \n", ulb)
print("\nu_max: \n", uub)
print("\nQ: \n", Q)
print("\nQN: \n", QN)
print("\nR: \n", R)
print("\nN: \n", N)
