import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
import numpy as np  
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import logging
logging.getLogger().setLevel(logging.INFO)

class Network(nn.Module):
    def __init__(self, n_features, hidden_list, n_outputs):
        super().__init__()

        self.n_features = n_features
        self.hidden_list = hidden_list
        self.n_outputs = n_outputs

        L = len(hidden_list)
        layers = []
        for i in range(L):
            if i == 0:
                fc = nn.Linear(n_features, self.hidden_list[i])
            elif i == L - 1:
                fc = nn.Linear(self.hidden_list[i-1], self.n_outputs)
            else:
                fc = nn.Linear(self.hidden_list[i-1], self.hidden_list[i])
        
            layers.append(fc)

        self.layers = nn.ModuleList(layers)

    
    def forward(self, x):
        x = x.astype(np.float)
        u = torch.from_numpy(x).float()
        for layer in self.layers:
            u = layer(u)
        # relu
        u_p = self.cvxpy_layer(u, x)
        print(u_p)


    def cvxpy_layer(self, parameter, state):
        x = state
        #n = u.shape[1]
        n = 1
        u_p = cp.Variable(n)

        E_m, f_m, G_m, h_m = self.get_constraint_matrices(x)
        u_m = parameter.detach().numpy()
        E = cp.Parameter((14))
        f = cp.Parameter((14))
        G = cp.Parameter((2))
        h = cp.Parameter((2))
        u = cp.Parameter((n))

        obj = cp.Minimize(cp.sum_squares(u-u_p))
        cons = [E @ u_p <= f, G @ u_p <= h]
        problem = cp.Problem(obj, cons)
        logging.info("  ---------- PROBLEM SETUP FINISHED ----------")

        layer = CvxpyLayer(problem, parameters = [E, f, G, h, u], variables = [u_p])

        E_m = torch.from_numpy(E_m).float()
        f_m = torch.from_numpy(f_m).float()
        G_m = torch.from_numpy(G_m).float()
        h_m = torch.from_numpy(h_m).float()
        u_m = torch.from_numpy(u_m).float()
        u_p_star, = layer(E_m, f_m, G_m, h_m, u_m)

        logging.info("  ---------- CVXPYLAYER CONSTRUCTED ----------")

        return u_p_star


    def get_constraint_matrices(self, x):
        """
            Eu^p <= f
            Gu^p <= h 
        """

        # constraints
        A = np.loadtxt('A.csv', delimiter=',')
        B = np.loadtxt('B.csv', delimiter=',')
        H = np.loadtxt('cinfH.csv', delimiter=',')
        C_c = H[:,0:2]
        d_c = H[:,2]
        C_u = np.array([1., -1.])
        d_u = np.array([0.5, -0.5])

        E = C_c @ B
        f = d_c - C_c @ A @ x
        G = C_u
        h = d_u

        return E, f, G, h

        


# input data
X = np.loadtxt('input_data.csv', delimiter=',')
x0 = X[0,:]

NN = Network(2, [8,4,2], 1)
NN.forward(x0)


#print("A dot B: \n", np.dot(A,B))
#print("A @ B: \n", A @ B)
"""
print("E: \n", E)
print("f: \n", f)
print("G: \n", G)
print("h: \n", h)
"""






