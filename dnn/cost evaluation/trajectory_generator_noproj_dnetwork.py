import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import cvxpy as cp
import numpy as np  
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import logging
logging.getLogger().setLevel(logging.INFO)
import statistics           
from sklearn.model_selection import train_test_split
import sys
import matplotlib.pyplot as plt
from polytope import Polytope
from datetime import datetime

example = str(sys.argv[1]).upper()
ntrajs = int(sys.argv[2])
nsim = int(sys.argv[3])
if len(sys.argv) > 4:
    date = sys.argv[4]
else:
    date = datetime.date(datetime.now())


filename = "ex{}/ex{}_".format(example, example)
filename_save = "ex{}/{}/trajectories/".format(example, date)

# (GLOBAL) network settings
#num_epochs = 200
batch_size = 1
learning_rate = 1e-4
device = 'cpu'
criterion = nn.MSELoss()


class Network(nn.Module):
    def __init__(self, hidden_list):
        super().__init__()

        self.n_features = 0
        self.n_outputs = 0
        self.hidden_list = hidden_list
        self.problem_params = {}

        # set optimization problem parameters (A, B, H, x_lb...)
        self.load_problem_parameters()

        L = len(hidden_list)
        layers = [nn.Linear(self.n_features, self.hidden_list[0])]
        for i in range(L):
            if i == L-1:
                # last relu layer
                fc = nn.Linear(self.hidden_list[i], self.n_outputs, bias=True)
            else:
                fc = nn.Linear(self.hidden_list[i], self.hidden_list[i+1], bias=True)
        
            layers.append(fc)

        self.layers = nn.ModuleList(layers)


    def load_problem_parameters(self):
        A = np.loadtxt(filename+'A.csv', delimiter=',')
        B = np.loadtxt(filename+'B.csv', delimiter=',')
        H = np.loadtxt(filename+'cinfH.csv', delimiter=',')
        xlb = np.loadtxt(filename+'xlb.csv', delimiter=',')
        xub =  np.loadtxt(filename+'xub.csv', delimiter=',')
        ulb = np.loadtxt(filename+'ulb.csv', delimiter=',')
        uub = np.loadtxt(filename+'uub.csv', delimiter=',')

        #print(A, "\n\n", B, "\n\n", H, "\n\n", x_lb, "\n\n", x_ub, "\n\n", u_lb, "\n\n", u_ub)

        n = A.shape[1]
        m = B.ndim
        num_constraints = H.shape[0]

        self.n_features = n
        self.n_outputs = m

        self.problem_params.update([ ('A', A), ('B', B), ('H', H), 
                                     ('xlb', xlb), ('xub', xub), ('ulb', ulb), ('uub', uub),
                                     ('n', n), ('m', m), ('num_constraints', num_constraints) ])


        logging.info("  ---------- PROBLEM PARAMETERS LOADED ----------")



    def forward(self, x):
        u = torch.from_numpy(x.numpy().astype(np.float)).float()
        
        for layer in self.layers[0:-1]:
            u = F.relu(layer(u))

        u = self.layers[-1](u)

        return u
        


if __name__ == "__main__":
    # define network
    print("\nRunning example: " + example + "\n")
    NN = Network([8, 8, 8, 8])
    NN.load_state_dict(torch.load('ex{}/{}/networks/ex{}_dnn_network_model.pt'.format(example, date, example)))
    optimizer = torch.optim.Adam(NN.parameters(), lr = learning_rate)
    NN.to(device)
    m, n = NN.problem_params['m'], NN.problem_params['n']

    # dynamics
    x0 = torch.from_numpy(np.array([-4.5, 2.])).float()
    A = NN.problem_params['A']
    B = NN.problem_params['B']
    H = NN.problem_params['H']
    xub, xlb = NN.problem_params['xub'], NN.problem_params['xlb']
    A_p, b_p = H[:,0:-1], H[:,-1]

    initial_states = np.loadtxt("ex{}/initial_states/ex{}_initial_states_{}.csv".format(example, example, ntrajs), delimiter=',')
    #initial_states = np.loadtxt(filename+"init_states_trajectories_ntrajs_{}_N_{}.csv".format(ntrajs, N), delimiter=',')
    
    s = 0
    for sample in initial_states:
        x0 = sample
        traj = [x0]
        traj_matrix = np.zeros((nsim,n))
        traj_matrix[0,:] = x0
        u_matrix = np.zeros((nsim-1,m))
        x0 = torch.from_numpy(sample).float()

        # simulate in closed loop
        for i in range(nsim-1):
            u = NN(x0)
            if m >= 2:
                x1 = np.dot(A, x0.numpy()) + np.dot(B, u.data.numpy().flatten())
            else:
                x1 = A @ x0.numpy() + B*u.data.numpy().flatten()
            
            traj.append(x1)
            traj_matrix[i+1,:] = x1
            u_matrix[i,:] = u.data.numpy()
            x0 = torch.from_numpy(x1).float()

        polytope = Polytope(A_p, b_p)
        if n <= 2:
            polytope.plot_poly(xlb, xub, infeasible_states = traj, show = False)

            title = "Example {} \n Sample: {} \n NoProj NN trajectory".format(example,s+1)
            xlabel = "$x_1$"
            ylabel = "$x_2$"
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel) 
            filen_fig = filename_save+"dnn_ntrajs_{}_nsim_{}_traj_{}".format(ntrajs, nsim, s+1)+".png"

            plt.savefig(filen_fig)
            #plt.show()
        
        np.savetxt(filename_save+'dnn_ntrajs_{}_nsim_{}_traj_{}'.format(ntrajs, nsim, s+1)+".csv", traj_matrix, delimiter=',')
        np.savetxt(filename_save+'dnn_controls_ntrajs_{}_nsim_{}_traj_{}'.format(ntrajs, nsim, s+1)+".csv", u_matrix, delimiter=',')
        s += 1

