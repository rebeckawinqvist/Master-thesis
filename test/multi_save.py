import numpy as np 

A = np.array([[1,2], [3,4]])
B = np.array([[5], [6]])

np.savez("matrices/matrices.npz", A=A, B=B)

matrices = np.load("matrices.npz")
print(matrices['A'])
print(matrices['B'])