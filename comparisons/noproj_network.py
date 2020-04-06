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

example = str(sys.argv[1]).upper()
nsamples_train = int(sys.argv[2])
nsamples_test = int(sys.argv[3])
nepochs = int(sys.argv[4])

example_name = 'ex'+example
folder_name = 'ex'+example+'/'
filename = folder_name+example_name+'_'

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
    NN = Network([8,8])
    #NN.load_state_dict(torch.load('saved_model.pt'), strict=False)
    optimizer = torch.optim.Adam(NN.parameters(), lr = learning_rate)
    NN.to(device)
    m = NN.problem_params['m']

    # data
    fin = 'ex{}/input_data/ex{}_'.format(example, example)
    fout = 'ex{}/output_data/ex{}_'.format(example, example)

    f_train_in, f_train_out = fin+'input_data_nsamples_{}_train.csv'.format(nsamples_train), fout+'output_data_nsamples_{}_train.csv'.format(nsamples_train)
    f_test_in, f_test_out = fin+'input_data_nsamples_{}_test.csv'.format(nsamples_test), fout+'output_data_nsamples_{}_test.csv'.format(nsamples_test)
    
    X_train = torch.from_numpy(np.loadtxt(f_train_in, delimiter=','))
    Y_train = torch.from_numpy(np.loadtxt(f_train_out, delimiter=',')[:,0:m])
    X_test = torch.from_numpy(np.loadtxt(f_test_in, delimiter=','))
    Y_test = torch.from_numpy(np.loadtxt(f_test_out, delimiter=',')[:,0:m])

    # split data into training set and test set
    # train, test = train_test_split(list(range(X.shape[0])), test_size=.25)

    # define dataset
    ds_train = TensorDataset(X_train, Y_train)
    ds_test = TensorDataset(X_test, Y_test)

    # define data loader
    train_set = DataLoader(ds_train, batch_size=batch_size_train)
    test_set = DataLoader(ds_test, batch_size=batch_size_test)

    to_train = True
    if to_train:
        # train the model
        epochs_losses = []
        for epoch in range(nepochs):
            batch_losses = []
            #i = 0
            for ix, (x, y) in enumerate(train_set):
                if y.shape[0] != batch_size_train:
                    continue
                
                _x = Variable(x).float()
                _y = Variable(y).float()

                # forward pass
                output = NN(_x)
                loss = criterion(output, _y)
                #loss.requires_grad = True

                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                #all_losses.append(loss.item())

            mbl = np.mean(np.sqrt(batch_losses))
            mbl = np.mean(batch_losses)
            epochs_losses.append(mbl)
            if epoch % 2 == 0:
                print("Epoch [{}/{}], Batch loss: {}".format(epoch, nepochs, mbl))

        logging.info("  ---------- TRAINING COMPLETED ----------")
        batch_size = batch_size_test
        epochs_arr = np.array(epochs_losses)

        """ UNCOMMENT TO SAVE MODEL"""
        torch.save(NN.state_dict(), 'ex{}/networks/ex{}_noproj_network_model_ntrain_{}.pt'.format(example, example, nsamples_train))
    else:
        NN.load_state_dict(torch.load('ex{}/networks/ex{}_noproj_network_model_ntrain_{}_nepochs_{}.pt'.format(example, example, nsamples_train, nepochs)))

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


    test_batch_losses = []
    relative_losses = []
    test_mse_losses = []
    test_nmse_losses = []
    true_values = []
    for ix, (x, y) in enumerate(test_set):
        _x = Variable(x).float()
        _y = Variable(y).float()

        if _y.shape[0] != batch_size_test:
            continue

        # forward pass
        test_output = NN(_x)
        test_loss = criterion(test_output, _y)

        test_mse_losses.append(test_loss.item())
        norm = torch.norm(_y, p=2)**2
        nmse_loss = test_loss.item()/norm
        test_nmse_losses.append(nmse_loss.item())
        true_values.append(norm.item())


        test_batch_losses.append(test_loss.item())

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
            #relative_losses.append(rel_loss)
            i += 1
        """
        #print("Batch loss: {}".format(test_loss.item()))

    logging.info("  ---------- TESTING COMPLETED ----------")

    mse_arr = np.array(test_mse_losses)
    nmse_arr = np.array(test_nmse_losses)
    true_values_arr = np.array(true_values)

    np.savetxt('ex{}/mse/ex{}_noproj_test_mse_losses_ntrain_{}_ntest_{}.csv'.format(example, example, nsamples_train, nsamples_test), mse_arr, delimiter=',')
    np.savetxt('ex{}/nmse/ex{}_noproj_test_nmse_losses_ntrain_{}_ntest_{}.csv'.format(example, example, nsamples_train, nsamples_test), nmse_arr, delimiter=',')
    #np.savetxt('ex{}/true_values/ex{}_noproj_true_values_ntrain_{}_ntest_{}.csv'.format(example,example,nsamples_train, nsamples_test), true_values_arr, delimiter=',')


    title = "Example {} \n Test cases".format(example)
    xlabel = "$x_1$"
    ylabel = "$x_2$"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    filen_fig = "ex{}/plots/ex{}_noproj_difficulty_points_ntrain_{}_ntest_{}_nepochs_{}.png".format(example, example, nsamples_train, nsamples_test, nepochs)
    plt.savefig(filen_fig)
    plt.show()
    print("Mean MSE loss: ", statistics.mean(test_mse_losses))
    print("Mean NMSE loss: ", statistics.mean(test_nmse_losses))

    # plot test losses
    x = [i+1 for i in range(len(test_mse_losses))]
    plt.plot(x, test_mse_losses, 'ro', linewidth=0.8, markersize=2)
    xlabel = "Test sample"
    ylabel = "MSE Loss"
    title = "Example {} \n Training samples: {},   Test samples: {} \n Epochs: {}".format(example, nsamples_train, nsamples_test, nepochs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 

    filen_fig = "ex{}/mse/ex{}_noproj_test_mse_losses_ntrain_{}_ntest_{}_nepochs_{}.png".format(example, example, nsamples_train, nsamples_test, nepochs)
    plt.savefig(filen_fig)
    plt.show()

    
    # plot test losses
    x = [i+1 for i in range(len(test_nmse_losses))]
    plt.plot(x, test_nmse_losses, 'ro', linewidth=0.8, markersize=2)
    xlabel = "Test sample"
    ylabel = "NMSE Loss"
    title = "Example {} \n Training samples: {}   Test samples: {} \n Epochs: {}".format(example, nsamples_train, nsamples_test, nepochs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 

    filen_fig = "ex{}/nmse/ex{}_noproj_test_nmse_losses_ntrain_{}_ntest_{}_nepochs_{}.png".format(example, example, nsamples_train, nsamples_test, nepochs)
    plt.savefig(filen_fig)
    plt.show()


    if to_train:
        # plot train losses
        np.savetxt('ex{}/train_losses/ex{}_noproj_train_losses_ntrain_{}_ntest_{}_nepochs_{}.csv'.format(example, example, nsamples_train, nsamples_test, nepochs), epochs_arr, delimiter=',')

        x = [i+1 for i in range(len(epochs_losses))]
        plt.plot(x, epochs_losses, 'ro', linewidth=0.8, markersize=2)
        xlabel = "Epoch"
        ylabel = "MSE loss"
        title = "Example {} \n Training samples: {}".format(example, nsamples_train)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel) 

        filen_fig = "ex{}/train_losses/ex{}_noproj_train_losses_ntrain_{}_ntest_{}_nepochs_{}.png".format(example, example, nsamples_train, nsamples_test, nepochs)
        plt.savefig(filen_fig)
        plt.show()

