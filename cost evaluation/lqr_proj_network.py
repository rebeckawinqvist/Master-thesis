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
from makedirs import *
from datetime import datetime 

example = str(sys.argv[1]).upper()
num_epochs = int(sys.argv[2])
if len(sys.argv) > 3:
    date = sys.argv[3]
else:
    date = datetime.date(datetime.now())

example_name = 'ex'+example
folder_name = 'ex'+example+'/'
filename = folder_name+example_name+'_'

makedirs(example, date)

# (GLOBAL) network settings
#num_epochs = 200
batch_size_train = 5
batch_size_test = 1
batch_size = batch_size_train

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
        m = self.problem_params['m']
        L = self.problem_params['L'].astype('float32') # LQR Gain
        
        for layer in self.layers[0:-1]:
            u = F.relu(layer(u))

        u = self.layers[-1](u)

        """
        # ====================================================================
        # ============= THIS MAY CAUSE AN ERROR FOR m > 1 ====================
        # ====================================================================

        u_tensor = torch.zeros([batch_size, m, 4])
        for i in range(batch_size):
            u_tensor[i] = u[i].expand(2,1)
        """
        u = u.unsqueeze(-1)
        #u.expand(batch_size,m,1)

        # projection/cvxpy-layer
        E, f, G, h, u = self.get_tensors(x, u)
        #print("A: \n", E, "\nb: ", f, "\n")
        #print("x: \n", x, "\nu: \n", u, "\n")
        #print(E@u <= f)
        if m < 2:
            u_lqr = torch.Tensor(x.data.numpy() @ -L).unsqueeze(-1).unsqueeze(-1)
        else:
            u_lqr = torch.Tensor(-L @ -x.unsqueeze(-1).data.numpy())

        u = u_lqr + u
        u, = self.cvxpy_layer(E, f, G, h, u, solver_args={'verbose': False, 'max_iters': 25000000})
        #u, = self.cvxpy_layer(G, h, u)

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
            f_tilde = dc - D_tilde @ states[i,:].data.numpy()
            f_tilde = np.expand_dims(f_tilde, axis=1)

            E[i] = torch.from_numpy(E_tilde).float()
            f[i] = torch.from_numpy(f_tilde).float()
            G[i] = torch.from_numpy(G_tilde).float()
            h[i] = torch.from_numpy(h_tilde).float()


        u = u_param

        # don't need to learn these, these are fixed
        #E.requires_grad = False
        #f.requires_grad = False
        #G.requires_grad = False
        #h.requires_grad = False
        #u.requires_grad = False

        return E, f, G, h, u
        

        

if __name__ == "__main__":
    # define network
    print("\nRunning example: " + example + "\n")
    NN = Network([8,8])
    #NN.load_state_dict(torch.load('saved_model.pt'), strict=False)
    optimizer = torch.optim.Adam(NN.parameters(), lr = learning_rate)
    NN.to(device)
    m = NN.problem_params['m']

    # data
    f_train_in, f_train_out = 'ex{}/input_data/ex{}_input_data_train.csv'.format(example, example), 'ex{}/output_data/ex{}_output_data_train.csv'.format(example, example)
    f_test_in, f_test_out = 'ex{}/input_data/ex{}_input_data_test.csv'.format(example, example), 'ex{}/output_data/ex{}_output_data_test.csv'.format(example, example)
    
    X_train = torch.from_numpy(np.loadtxt(f_train_in, delimiter=','))
    Y_train = torch.from_numpy(np.loadtxt(f_train_out, delimiter=',')[:,0:m])
    X_test = torch.from_numpy(np.loadtxt(f_test_in, delimiter=','))
    Y_test = torch.from_numpy(np.loadtxt(f_test_out, delimiter=',')[:,0:m])

    # split data into training set and test set
    #train, test = train_test_split(list(range(X.shape[0])), test_size=.25)
    
    # define dataset
    ds_train = TensorDataset(X_train, Y_train)
    ds_test = TensorDataset(X_test, Y_test)

    # define data loader
    train_set = DataLoader(ds_train, batch_size=batch_size_train)
    test_set = DataLoader(ds_test, batch_size=batch_size_test)


    # train the model
    epochs_losses = []
    for epoch in range(num_epochs):
        batch_losses = []
        #i = 0
        for ix, (x, y) in enumerate(train_set):
            if y.shape[0] != batch_size_train:
                continue
            
            _x = Variable(x).float()
            _y = Variable(y).float()
            _y = _y.unsqueeze(-1)
            _y.expand(batch_size,m,1)

            """
            y = np.expand_dims(y, axis=1)
            print(y)
            y_t = torch.zeros([batch_size, NN.problem_params['m'], 1])
            for i in range(batch_size):
                y_t[i] = torch.from_numpy(np.expand_dims(y[i], axis=1))
                """

            #_y = Variable(y_t).float()

            # forward pass
            output = NN(_x)
            loss = criterion(output, _y)
            #loss.requires_grad = True

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """
            for p in NN.parameters():
                if p.grad is not None:
                    print(p)
            """


            batch_losses.append(loss.item())
            #all_losses.append(loss.item())

        mbl = np.mean(np.sqrt(batch_losses))
        mbl = np.mean(batch_losses)
        epochs_losses.append(mbl)
        if epoch % 2 == 0:
            print("Epoch [{}/{}], Batch loss: {}".format(epoch, num_epochs, mbl))

    logging.info("  ---------- TRAINING COMPLETED ----------")
    
    """ UNCOMMENT TO SAVE MODEL """
    torch.save(NN.state_dict(), 'ex{}/{}/networks/ex{}_lqr_proj_network_model.pt'.format(example, date, example))
    
    # test the model
    logging.info("  ---------- TESTING STARTED ----------")
    batch_size = batch_size_test

    # plot if n = 2
    n = NN.problem_params['n']
    if n == 2:
        xlb, xub = NN.problem_params['xlb'], NN.problem_params['xub']
        x_start, x_stop = xlb[0], xub[0]
        y_start, y_stop = xlb[1], xub[1]
        H = NN.problem_params['H']
        numOfCons = H.shape[0]

        plt.axis([x_start-1,x_stop+1,y_start-1,y_stop+1])
        x = np.linspace(x_start,x_stop,1000)
        for i in range(numOfCons):
            if H[i,1] != 0:
                k = -H[i,0]/H[i,1]
                m = H[i,2]/H[i,1]
                f = k * x + m
                plt.plot(x,f, '-k', linewidth=0.8, color='gainsboro')
            else:
                plt.axvline(x = H[i,2]/H[i,0], ymin=y_start, ymax=y_stop, linewidth=0.8, color='gainsboro')


    test_mse_losses = []
    true_values = []
    for ix, (x, y) in enumerate(test_set):
        _x = Variable(x).float()
        _y = Variable(y).float()
        _y = _y.unsqueeze(-1)
        #_y.expand(batch_size,m,1)
        if _y.shape[0] != batch_size_test:
            continue
        """
        y = np.expand_dims(y, axis=1)
        y_t = torch.zeros([batch_size, NN.problem_params['m'], 1])
        for i in range(batch_size):
            y_t[i] = torch.from_numpy(np.expand_dims(y[i], axis=1))

        _y = Variable(y_t).float()
        """

        # forward pass
        test_output = NN(_x)
        test_loss = criterion(test_output, _y)

        test_mse_losses.append(test_loss.item())
        norm = torch.norm(_y, p=2)**2
        #nmse_loss = test_loss.item()/norm
        #test_nmse_losses.append(nmse_loss.item())
        true_values.append(norm.item())

        """
        diff = (_y - test_output).detach().numpy()
        abs_diff = np.absolute(diff)
        rel_losses = np.divide(abs_diff,np.absolute(_y))

        i = 0
        for rel_loss in rel_losses:
            if n == 2:
                state = _x[i].numpy()
                if abs(rel_loss) > 0.5:
                    plt.plot(state[0], state[1], 'ro', linewidth=0.8, markersize=2)
                    #print(test_output[i].data, _y[i].data)
                else:
                    plt.plot(state[0], state[1], 'go', linewidth=0.8, markersize=2)

            test_nmse_losses.append(rel_loss.data.numpy().flatten()[0])
            i += 1
        """

        #print("Batch loss: {}".format(test_loss.item()))


    logging.info("  ---------- TESTING COMPLETED ----------")

    mse_arr = np.array(test_mse_losses)
    #nmse_arr = np.array(test_nmse_losses)
    epochs_arr = np.array(epochs_losses)
    true_values_arr = np.array(true_values)

    filename_test = 'ex{}/{}/test_losses/ex{}_'.format(example, date, example)
    filename_train = 'ex{}/{}/train_losses/ex{}_'.format(example, date, example)

    np.savetxt(filename_test+'lqr_proj_test_mse_losses.csv', mse_arr, delimiter=',')
    np.savetxt(filename_train+'lqr_proj_train_losses.csv', epochs_arr, delimiter=',')
    np.savetxt('ex{}/{}/true_values/ex{}_lqr_true_values.csv'.format(example, date, example), true_values_arr, delimiter=',')


    title = "Example {} \n Test cases".format(example)
    xlabel = "$x_1$"
    ylabel = "$x_2$"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    filen_fig = "ex{}/{}/plots/ex{}_lqr_proj_difficulty_points.png".format(example, date, example)
    plt.savefig(filen_fig)
    plt.show()


    # plot train losses
    x = [i+1 for i in range(len(epochs_losses))]
    plt.plot(x, epochs_losses, 'ro', linewidth=0.8, markersize=2)
    xlabel = "Epoch"
    ylabel = "MSE loss"
    title = "Example {}".format(example)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 

    filen_fig = "ex{}/{}/train_losses/ex{}_lqr_proj_train_losses.png".format(example, date, example)
    plt.savefig(filen_fig)
    plt.show()


    # plot test losses
    x = [i+1 for i in range(len(test_mse_losses))]
    plt.plot(x, test_mse_losses, 'ro', linewidth=0.8, markersize=2)
    xlabel = "Test sample"
    ylabel = "MSE Loss"
    title = "Example {}".format(example)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 

    filen_fig = "ex{}/{}/test_losses/ex{}_lqr_proj_test_mse_losses.png".format(example, date, example)
    plt.savefig(filen_fig)
    plt.show()

    """
    # plot test losses
    x = [i+1 for i in range(len(test_nmse_losses))]
    plt.plot(x, test_nmse_losses, 'ro', linewidth=0.8, markersize=2)
    xlabel = "Test sample"
    ylabel = "NMSE Loss"
    title = "Example {}".format(example)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 

    filen_fig = filename+"lqr_proj_test_nmse_losses.png"
    plt.savefig(filen_fig)
    plt.show()
    """


