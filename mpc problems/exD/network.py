import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
import numpy as np  
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import logging
logging.getLogger().setLevel(logging.INFO)
import statistics

class Network(nn.Module):
    def __init__(self, n_features, hidden_list, n_outputs):
        super().__init__()

        self.n_features = n_features
        self.hidden_list = hidden_list
        self.n_outputs = n_outputs

        L = len(hidden_list)
        layers = [nn.Linear(n_features, self.hidden_list[0])]
        for i in range(L):
            if i == L-1:
                # last relu layer
                fc = nn.Linear(self.hidden_list[i], n_outputs)
            else:
                fc = nn.Linear(self.hidden_list[i], self.hidden_list[i+1])
        
            layers.append(fc)

        self.reluLayers = nn.ModuleList(layers)

        # cvxpy-layer
        self.cvxpy_layer = self.create_cvxpylayer()


    
    def create_cvxpylayer(self):
        n = 1

        u_p = cp.Variable(n)
        E = cp.Parameter((14))
        f = cp.Parameter((14))
        G = cp.Parameter((2))
        h = cp.Parameter((2))
        u = cp.Parameter((n))

        obj = cp.Minimize(cp.sum_squares(u-u_p))
        cons = [E @ u_p <= f, G @ u_p <= h]
        problem = cp.Problem(obj, cons)
        assert problem.is_dpp()
        assert problem.is_dcp()

        layer = CvxpyLayer(problem, parameters = [E, f, G, h, u], variables = [u_p])
        logging.info("  ---------- CVXPY-LAYER CONSTRUCTED ----------")

        return layer


    def forward(self, x):
        x = x.astype(np.float)
        u = torch.from_numpy(x).float()

        for layer in self.reluLayers:
            u = F.relu(layer(u))

        # projection/cvxpy-layer
        E, f, G, h, u = self.get_constraints(x, u)
        """
        for i in range(14):
            if np.array(E)[i] >= 0:
                print("u <= ", (np.array(f)[i])/(np.array(E)[i]))
            else:
                print("u >= ", (np.array(f)[i])/(np.array(E)[i]))
            #print(np.array(E)[i], np.array(f)[i])
        print(G, "\n", h)
        print("\n\n\n")
        """
        
        u_p = self.cvxpy_layer(E, f, G, h, u)
        #u_p = self.cvxpy_layer(G, h, u)

        return u_p


    def get_constraints(self, state, u_param):
        """
            Eu^p <= f
            Gu^p <= h 
        """

        # constraints
        A = np.loadtxt('exD_A.csv', delimiter=',')
        B = np.loadtxt('exD_B.csv', delimiter=',')
        H = np.loadtxt('exD_cinfH.csv', delimiter=',')
        u_low = -0.5
        u_high = 0.5
        C_c = H[:,0:2]
        d_c = H[:,2]
        C_u = np.array([1., -1.])
        d_u = np.array([u_high, -u_low])

        E = C_c @ B
        f = d_c - C_c @ A @ state
        G = C_u
        h = d_u
        u = u_param.detach().numpy()

        E = torch.from_numpy(E).float()
        f = torch.from_numpy(f).float()
        G = torch.from_numpy(G).float()
        h = torch.from_numpy(h).float()
        u = torch.from_numpy(u).float()

        # don't need to learn these, these are fixed
        E.requires_grad = False
        f.requires_grad = False
        G.requires_grad = False
        h.requires_grad = False
        u.requires_grad = False

        return E, f, G, h, u
        

        

if __name__ == "__main__":
    # data
    X = np.loadtxt('exD_input_data_grid.csv', delimiter=',')
    Y = np.loadtxt('exD_output_data_grid.csv', delimiter=',')
    print("Y: ", Y)

    NN = Network(2, [8,8], 1)
    print("Network: \n", NN)

    # create an optimizer
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(NN.parameters(), lr = learning_rate)
    # create a loss function
    criterion = nn.MSELoss()

    num_epochs = 5
    device = 'cpu'

    NN.to(device)


    # train the model
    for epoch in range(num_epochs):
        i = 0
        for row in X:
            # forward pass
            output, = NN(row)
            target = torch.from_numpy(np.array([Y[i,0]])).float()
            print(row, output, target)
            loss = criterion(output, target)
            loss.requires_grad = True

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1

    logging.info("  ---------- TRAINING COMPLETED ----------")
    

    # test the model
    loss_list = []
    with torch.no_grad():
        i = 0        
        for row in X:
            output, = NN(row)
            target = torch.from_numpy(np.array([Y[i,0]])).float()
            loss = criterion(output, target)
            #print(loss.item())
            loss_list.append(loss.item())
            i += 1

    print("Mean mse: ", statistics.mean(loss_list))  

