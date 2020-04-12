import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
import sys
import logging
import matplotlib.pyplot as plt
from makedirs import *
from datetime import datetime

logging.getLogger().setLevel(logging.INFO)
location = os.getcwd()

#example = str(sys.argv[1]).upper()
nsamples_train = int(sys.argv[1])
nsamples_test = int(sys.argv[2])
nepochs = int(sys.argv[3])
if len(sys.argv) > 4:
    date = sys.argv[4]
else:
    date = datetime.date(datetime.now())

makedirs(date)

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
    n, m = 2, 1
    #n = np.loadtxt('ex{}/ex{}_A.csv'.format(example, example), delimiter=',').shape[1]
    #m = np.loadtxt('ex{}/ex{}_B.csv'.format(example, example), delimiter=',').ndim

    hidden_size = 10
    nlayers = 1
    rnn = RNN(n, m, hidden_size, nlayers)
    optimizer = torch.optim.Adam(rnn.parameters(), lr = learning_rate)
    loss_fn = nn.MSELoss()

    # define datasets
    loc_trajs = "{}\\{}\\trajectories".format(location, date)
    train_trajectories, test_trajectories = os.listdir("{}\\train".format(loc_trajs)), os.listdir("{}\\test".format(loc_trajs))

    train_sets, test_sets = [], []

    for traj in train_trajectories:
        x_train = torch.from_numpy(np.loadtxt("{}\\train\\{}".format(loc_trajs, traj), delimiter=','))[:,0:n]
        y_train = torch.from_numpy(np.loadtxt("{}\\train\\{}".format(loc_trajs, traj), delimiter=','))[:,n:]
        
        train_set = SeqDataset(x_train,  y_train, seq_dim, n, m)
        train_sets.append(train_set)    


    for traj in test_trajectories:
        x_test = torch.from_numpy(np.loadtxt("{}\\test\\{}".format(loc_trajs, traj), delimiter=','))[:,0:n]
        y_test = torch.from_numpy(np.loadtxt("{}\\test\\{}".format(loc_trajs, traj), delimiter=','))[:,n:]
    
        test_set = SeqDataset(x_test,  y_test, seq_dim, n, m)
        test_sets.append(test_set)  


    ds_train, ds_test = ConcatDataset(train_sets), ConcatDataset(test_sets)
    train_set, test_set = DataLoader(ds_train, batch_size=batch_size_train, drop_last=True), DataLoader(ds_test, batch_size=batch_size_test, drop_last=True)


    logging.info("  ---------- TRAINING STARTED ----------")
    epochs_losses = []
    for epoch in range(nepochs):
        batch_losses = []
        for ix, (x, y) in enumerate(train_set):
            _x = Variable(x)
            _y = Variable(y)

            output = rnn(_x)
            loss = loss_fn(output, _y[:, -1, :])
            
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

    # torch.save(rnn.state_dict(), '{}/rnn_model.pt'.format(date))

    # Test model
    logging.info("  ---------- TESTING STARTED ----------")
    batch_size = batch_size_test
    
    test_mse_losses = []
    true_values = []
    
    for ix, (x, y) in enumerate(test_set):
        _x = Variable(x)
        _y = Variable(y)

        test_output = rnn(_x)
        test_loss = loss_fn(test_output, _y[:, -1, :])

        test_mse_losses.append(test_loss.item())
        norm = torch.norm(_y, p=2)**2
        true_values.append(norm.item())


    logging.info("  ---------- TESTING COMPLETED ----------")

    mse_arr = np.array(test_mse_losses)
    true_values_arr = np.array(true_values)

    np.savetxt('{}/test_losses/rnn_test_mse_losses.csv'.format(date), mse_arr, delimiter=',')
    np.savetxt('{}/true_values/rnn_true_values.csv'.format(date), true_values_arr, delimiter=',')
    np.savetxt('{}/train_losses/rnn_train_losses.csv'.format(date), epochs_arr, delimiter=',')

    
    # plot train losses
    x = [i+1 for i in range(len(epochs_losses))]
    plt.plot(x, epochs_losses, 'ro', linewidth=0.8, markersize=2)
    xlabel = "Epoch"
    ylabel = "MSE loss"
    title = "Epoch losses"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 

    filen_fig = "{}/train_losses/rnn_train_losses.png".format(date)
    plt.savefig(filen_fig)
    plt.show()


    # plot test losses
    x = [i+1 for i in range(len(test_mse_losses))]
    plt.plot(x, test_mse_losses, 'ro', linewidth=0.8, markersize=2)
    xlabel = "Test sample"
    ylabel = "MSE Loss"
    title = "Test losses"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 

    filen_fig = "{}/test_losses/rnn_test_mse_losses.png".format(date)
    plt.savefig(filen_fig)
    plt.show()