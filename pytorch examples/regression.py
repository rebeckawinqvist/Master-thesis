import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#X, y = load_boston(return_X_y=True)
data_generation = 'grid'
filename = 'exA_'
f_in, f_out = filename+'input_data_'+data_generation+'.csv', filename+'output_data_'+data_generation+'.csv'
X = torch.from_numpy(np.loadtxt(f_in, delimiter=','))
y = torch.from_numpy(np.loadtxt(f_out, delimiter=',')[:,0:1])

# create train and test indices
train, test = train_test_split(list(range(X.shape[0])), test_size=.25)

input_size = 2
hidden_layer_size = 8
learning_rate = 0.05
batch_size = 50
num_epochs = 100

"""
class PrepareData(Dataset):

    def __init__(self, X, y, scale_X=True):
        if not torch.is_tensor(X):
            if scale_X:
                X = StandardScaler().fit_transform(X)
                self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
"""

#ds = PrepareData(X, y=y, scale_X=True)
ds = TensorDataset(X,y)

train_set = DataLoader(ds, batch_size=batch_size,
                       sampler=SubsetRandomSampler(train))
test_set = DataLoader(ds, batch_size=batch_size,
                      sampler=SubsetRandomSampler(test))

class RegressionModel(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RegressionModel, self).__init__()
        self.dense_h1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu_h1 = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.5)
        self.dense_out = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, X):

        out = self.relu_h1(self.dense_h1(X))
        #out = self.dropout(out)
        out = self.relu_h1(self.dense_out(out))


        return out

m = RegressionModel(input_size=input_size, hidden_size=hidden_layer_size)

cost_func = nn.MSELoss()
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

all_losses = []
for e in range(num_epochs):
    batch_losses = []

    for ix, (Xb, yb) in enumerate(train_set):

        _X = Variable(Xb).float()
        _y = Variable(yb).float()

        #==========Forward pass===============

        preds = m(_X)
        loss = cost_func(preds, _y)

        #==========backward pass==============

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        all_losses.append(loss.item())

    mbl = np.mean(np.sqrt(batch_losses)).round(3)

    if e % 5 == 0:
        print("Epoch [{}/{}], Batch loss: {}".format(e, num_epochs, mbl))

# prepares model for inference when trained with a dropout layer
#print(m.training)
#m.eval()
#print(m.training)

test_batch_losses = []
for _X, _y in test_set:

    _X = Variable(_X).float()
    _y = Variable(_y).float()

    #apply model
    test_preds = m(_X)
    test_loss = cost_func(test_preds, _y)

    test_batch_losses.append(test_loss.item())
    print("Batch loss: {}".format(test_loss.item()))