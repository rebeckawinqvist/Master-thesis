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

example = str(sys.argv[1]).upper()
ntrajs = int(sys.argv[2])
N = int(sys.argv[3])

example_name = 'ex'+example
folder_name = 'ex'+example+'/'
filename = folder_name+example_name+'_'
filename_save = 'ex'+example+'/trajectories/'

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

        # cvxpy-layer
        self.cvxpy_layer = self.create_cvxpylayer()


    def load_problem_parameters(self):
        A = np.loadtxt(filename+'A.csv', delimiter=',')
        B = np.loadtxt(filename+'B.csv', delimiter=',')
        H = np.loadtxt(filename+'cinfH.csv', delimiter=',')
        L = np.loadtxt(filename+'L.csv', delimiter=',')
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

        self.problem_params.update([ ('A', A), ('B', B), ('H', H), ('L', L),
                                     ('xlb', xlb), ('xub', xub), ('ulb', ulb), ('uub', uub),
                                     ('n', n), ('m', m), ('num_constraints', num_constraints) ])


        logging.info("  ---------- PROBLEM PARAMETERS LOADED ----------")


    
    def create_cvxpylayer(self):
        m = self.problem_params['m']
        num_constraints = self.problem_params['num_constraints']        

        u_p = cp.Variable((m, 1))
        E = cp.Parameter((num_constraints, m))
        f = cp.Parameter((num_constraints, 1))
        G = cp.Parameter((2*m, m))
        h = cp.Parameter((2*m, 1))
        u = cp.Parameter((m, 1))   

        E.requires_grad = False
        f.requires_grad = False
        G.requires_grad = False
        h.requires_grad = False
        u.requires_grad = False   

        obj = cp.Minimize(cp.sum_squares(u-u_p))
        cons = [E @ u_p <= f, G @ u_p <= h]
        #cons = [G @ u_p <= h]
        problem = cp.Problem(obj, cons)
        assert problem.is_dpp() 
        assert problem.is_dcp()

        layer = CvxpyLayer(problem, parameters = [E, f, G, h, u], variables = [u_p])
        #layer = CvxpyLayer(problem, parameters = [G, h, u], variables = [u_p])
        logging.info("  ---------- CVXPY-LAYER CONSTRUCTED ----------")

        return layer


    def forward(self, x):
        u = torch.from_numpy(x.numpy().astype(np.float)).float()
        L = self.problem_params['L'].astype('float32') # LQR Gain

        for layer in self.layers[0:-1]:
            u = F.relu(layer(u))

        u = self.layers[-1](u)

        u = u.unsqueeze(-1).unsqueeze(0)
        #print(u.unsqueeze(0))    
        # projection/cvxpy-layer
        E, f, G, h, u = self.get_tensors(x, u)

        if m < 2:
            if batch_size == 1:
                u_lqr = torch.Tensor(np.array([-L @ x.data.numpy()])).unsqueeze(-1).unsqueeze(-1)
            else:
                u_lqr = torch.Tensor(-L @ x.data.numpy()).unsqueeze(-1).unsqueeze(-1)    
        else:
            if batch_size == 1:
                u_lqr = torch.Tensor(-L @ x.data.numpy()).unsqueeze(-1).unsqueeze(0)
            else:
                pass
                #u_lqr = -L @ x

        u = u_lqr + u
        u, = self.cvxpy_layer(E, f, G, h, u, solver_args={'verbose': False, 'max_iters': 4000000})

        return u


    def get_tensors(self, states, u_param):
        """
            Eu^p <= f
            Gu^p <= h 
        """

        # batch size (last training/test batch is sometimes smaller than set batch size)
        bs = u_param.shape[0]
        # constraints
        m = self.problem_params['m']
        num_constraints = self.problem_params['num_constraints']

        A = self.problem_params['A']
        B = self.problem_params['B']
        H = self.problem_params['H']
        ulb = self.problem_params['ulb']
        uub = self.problem_params['uub']
        if m > 1:
            ulb = np.expand_dims(ulb, axis=1)
            uub = np.expand_dims(uub, axis=1)

        Cc = H[:,0:-1]
        dc = H[:,-1]
        #Cu = np.kron(np.eye(m),np.array([[1.], [-1.]]))
        Cu = np.vstack((np.eye(m), -np.eye(m)))
        du = np.vstack((uub, -ulb))

        D_tilde = Cc @ A
        E_tilde = Cc @ B
        G_tilde = Cu
        h_tilde = du

        if m == 1:
            E_tilde = np.expand_dims(E_tilde, axis=1)

        E = torch.zeros([batch_size, num_constraints, m])
        f = torch.zeros([batch_size, num_constraints, 1])
        G = torch.zeros([batch_size, 2*m, m])
        h = torch.zeros([batch_size, 2*m, 1])


        for i in range(bs):
            f_tilde = dc - D_tilde @ states.data.numpy()
            f_tilde = np.expand_dims(f_tilde, axis=1)

            E[i] = torch.from_numpy(E_tilde).float()
            f[i] = torch.from_numpy(f_tilde).float()
            G[i] = torch.from_numpy(G_tilde).float()
            h[i] = torch.from_numpy(h_tilde).float()

        u = u_param

        return E, f, G, h, u
        


if __name__ == "__main__":
    # define network
    print("\nRunning example: " + example + "\n")
    NN = Network([8,8])
    NN.load_state_dict(torch.load('ex{}/networks/ex{}_lqr_proj_network_model.pt'.format(example, example)))
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
        traj_matrix = np.zeros((N,n))
        traj_matrix[0,:] = x0
        u_matrix = np.zeros((N-1,m))
        x0 = torch.from_numpy(sample).float()

        # simulate in closed loop
        for i in range(N-1):
            u = NN(x0)
            if m >= 2:
                x1 = np.dot(A, x0.numpy()) + np.dot(B, u.data.numpy().flatten())
            else:
                x1 = A @ x0.numpy() + B*u.data.numpy().flatten()
            
            traj.append(x1)
            traj_matrix[i+1,:] = x1
            u_matrix[i,:] = u.data.numpy().flatten()
            x0 = torch.from_numpy(x1).float()


        polytope = Polytope(A_p, b_p)
        if n <= 2 and traj:
            polytope.plot_poly(xlb, xub, infeasible_states = traj, show = False)

            title = "Example {}: \n Sample: {} \n LQR Proj NN trajectory".format(example,s+1)
            xlabel = "$x_1$"
            ylabel = "$x_2$"
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel) 
            filen_fig = filename_save+"lqr_projNN_ntrajs_{}_N_{}_traj_{}".format(ntrajs, N, s+1)+".png"

            plt.savefig(filen_fig)
            #plt.show()

        np.savetxt(filename_save+'lqr_projNN_ntrajs_{}_N_{}_traj_{}'.format(ntrajs, N, s+1)+".csv", traj_matrix, delimiter=',')
        np.savetxt(filename_save+'lqr_projNN_controls_ntrajs_{}_N_{}_traj_{}'.format(ntrajs, N, s+1)+".csv", u_matrix, delimiter=',')
        s += 1

