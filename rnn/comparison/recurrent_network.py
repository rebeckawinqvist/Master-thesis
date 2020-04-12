import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset
import sys
import logging
import matplotlib.pyplot as plt
from makedirs import *
from datetime import datetime

logging.getLogger().setLevel(logging.INFO)

example = str(sys.argv[1]).upper()
nsamples_train = int(sys.argv[2])
nsamples_test = int(sys.argv[3])
nepochs = int(sys.argv[4])
if len(sys.argv) > 5:
    date = sys.argv[5]
else:
    date = datetime.date(datetime.now())

makedirs(example, date)

# Global network settings
batch_size_train = 5
batch_size_test = 1
batch_size = batch_size_train

seq_dim = 5

learning_rate = 1e-4
device = 'cpu'
criterion = nn.MSELoss()


# customized data set
class SeqDataset(Dataset):
    def __init__(self, x, y, seq_dim, n, m):
        self.seq_dim = seq_dim
        self.x = x
        self.y = y
        self.n = n
        self.m = m
        
    def __len__(self):
        #return int(self.x.shape[0]) // int(self.seq_dim)
        return self.x.shape[0] - self.seq_dim + 1

    def __getitem__(self, idx):
        xout = torch.zeros(self.seq_dim, self.n)
        yout = torch.zeros(self.seq_dim, self.m)
        for i in range(0, self.seq_dim):
            xout[i] = self.x[idx + i]
            yout[i] = self.y[idx + i]   
        return xout, yout 


class RNN(nn.Module):
    def __init__(self, n, m, hidden_size, nlayers):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(input_size = n,
                            hidden_size = hidden_size,
                            num_layers = nlayers,
                            batch_first = True)

        self.output_layer = nn.Linear(hidden_size, m)

    
    def forward(self, x):
        u, (h_n, c_n) = self.lstm(x)

        u = self.output_layer(u[:, -1, :]) # return only u from last time step
        return u



if __name__ == "__main__":
    # define network
    print("Running example: {}".format(example))
    n = np.loadtxt('ex{}/ex{}_A.csv'.format(example, example), delimiter=',').shape[1]
    m = np.loadtxt('ex{}/ex{}_B.csv'.format(example, example), delimiter=',').ndim

    hidden_size = 10
    nlayers = 1
    rnn = RNN(n, m, hidden_size, nlayers)
    optimizer = torch.optim.Adam(RNN.parameters(), lr = learning_rate)
    loss_fn = nn.MSELoss()

    # data
    fin = 'ex{}/input_data/ex{}_'.format(example, example)
    fout = 'ex{}/output_data/ex{}_'.format(example, example)

    f_train_in, f_train_out = fin+'input_data_nsamples_{}_train.csv'.format(nsamples_train), fout+'output_data_nsamples_{}_train.csv'.format(nsamples_train)
    f_test_in, f_test_out = fin+'input_data_nsamples_{}_test.csv'.format(nsamples_test), fout+'output_data_nsamples_{}_test.csv'.format(nsamples_test)
    
    X_train = torch.from_numpy(np.loadtxt(f_train_in, delimiter=','))
    Y_train = torch.from_numpy(np.loadtxt(f_train_out, delimiter=',')[:,0:m])
    X_test = torch.from_numpy(np.loadtxt(f_test_in, delimiter=','))
    Y_test = torch.from_numpy(np.loadtxt(f_test_out, delimiter=',')[:,0:m])

    # define dataset
    ds_train = SeqDataset(X_train, Y_train, seq_dim, n, m)
    ds_test = SeqDataset(X_test, Y_test, seq_dim, n, m)

    # define data loader
    train_set = DataLoader(ds_train, batch_size=batch_size_train, drop_last = True)
    test_set = DataLoader(ds_test, batch_size=batch_size_test, drop_last = True)


    logging.info("  ---------- TRAINING STARTED ----------")
    epochs_losses = []
    for epoch in range(nepochs):
        batch_losses = []
        for ix, (x, y) in enumerate(train_set):
            _x = Variable(x)
            _y = Variable(y)

            output = rnn(_x)
            loss = loss_fn(output, _y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        mbl = np.mean(batch_losses)
        epochs_losses.append(mbl)
        if epoch % 2 == 0:
            print("Epoch [{}/{}], Batch loss: {}".format(epoch, nepochs, mbl))

    logging.info("  ---------- TRAINING COMPLETED ----------")
    epochs_arr = np.array(epochs_losses)

    """ UNCOMMENT TO SAVE MODEL """
    torch.save(rnn.state_dict(), 'ex{}/{}/ex{}_rnn_model_ntrain_{}.pt'.format(example, date, example, nsamples_train))

    # Test model
    logging.info("  ---------- TESTING STARTED ----------")
    batch_size = batch_size_test

    
    test_mse_losses = []
    true_values = []
    
    for ix, (x, y) in enumerate(test_set):
        x = Variable(x)
        _y = Variable(y)

        test_output = rnn(_x)
        test_loss = loss_fn(test_output, _y)

        test_mse_losses.append(test_loss.item())
        norm = torch.norm(_y, p=2)**2
        true_values.append(norm.item())


    logging.info("  ---------- TESTING COMPLETED ----------")

    mse_arr = np.array(test_mse_losses)
    true_values_arr = np.array(true_values)

    np.savetxt('ex{}/{}/mse/ex{}_test_mse_losses_ntrain_{}_ntest_{}.csv'.format(example, date, example, nsamples_train, nsamples_test), mse_arr, delimiter=',')
    np.savetxt('ex{}/{}/true_values/ex{}_true_values_ntrain_{}_ntest_{}.csv'.format(example, date, example, nsamples_train, nsamples_test), true_values_arr, delimiter=',')

    
    # plot train losses
    x = [i+1 for i in range(len(epochs_losses))]
    plt.plot(x, epochs_losses, 'ro', linewidth=0.8, markersize=2)
    xlabel = "Epoch"
    ylabel = "MSE loss"
    title = "Example {} \n Training samples: {}".format(example, nsamples_train)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 

    filen_fig = "ex{}/{}/train_losses/ex{}_train_losses_ntrain_{}_ntest_{}_nepochs_{}.png".format(example, date, example, nsamples_train, nsamples_test, nepochs)
    #plt.savefig(filen_fig)
    plt.show()
    
    # plot test losses
    x = [i+1 for i in range(len(test_mse_losses))]
    plt.plot(x, test_mse_losses, 'ro', linewidth=0.8, markersize=2)
    xlabel = "Test sample"
    ylabel = "MSE Loss"
    title = "Example {} \n Training samples: {},   Test samples: {} \n Epochs: {}".format(example, nsamples_train, nsamples_test, nepochs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 

    filen_fig = "ex{}/{}/mse/ex{}_test_mse_losses_ntrain_{}_ntest_{}_nepochs_{}.png".format(example, date, example, nsamples_train, nsamples_test, nepochs)
    plt.savefig(filen_fig)
    plt.show()
            


