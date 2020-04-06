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

filename_mse = 'mse/exE_'
filename_tv = 'true_values/exE_' 
ntest = 500

trains = [200, 500, 1000, 1500]
example = "E"

i = 1
for num in trains:
    proj_mse = np.loadtxt(filename_mse+'test_mse_losses_ntrain_{}_ntest_{}.csv'.format(num, ntest), delimiter=',')[1:]
    noproj_mse = np.loadtxt(filename_mse+'noproj_test_mse_losses_ntrain_{}_ntest_{}.csv'.format(num, ntest), delimiter=',')[1:]
    lqr_mse = np.loadtxt(filename_mse+'lqr_test_mse_losses_ntrain_{}_ntest_{}.csv'.format(num, ntest), delimiter=',')[1:]

    proj_tv = np.loadtxt(filename_tv+"true_values_ntrain_{}_ntest_{}.csv".format(num, ntest), delimiter=',')[1:]
    noproj_tv = np.loadtxt(filename_tv+"noproj_true_values_ntrain_{}_ntest_{}.csv".format(num, ntest), delimiter=',')[1:]
    lqr_tv = np.loadtxt(filename_tv+"lqr_true_values_ntrain_{}_ntest_{}.csv".format(num, ntest), delimiter=',')[1:]

    proj_nmse = np.zeros(proj_mse.shape)
    noproj_nmse = np.zeros(proj_mse.shape)
    lqr_nmse = np.zeros(proj_mse.shape)

    for i in range(len(proj_mse)):
        proj = proj_mse[i]/proj_tv[i]
        noproj = noproj_mse[i]/noproj_tv[i]
        lqr = lqr_mse[i]/lqr_tv[i]

        proj_nmse[i] = proj
        noproj_nmse[i] = noproj
        lqr_nmse[i] = lqr

    
    np.savetxt('nmse/ex{}_test_nmse_losses_ntrain_{}_ntest_{}.csv'.format(example, num, ntest), proj_nmse, delimiter=',')
    np.savetxt('nmse/ex{}_lqr_test_nmse_losses_ntrain_{}_ntest_{}.csv'.format(example, num, ntest), lqr_nmse, delimiter=',')
    np.savetxt('nmse/ex{}_noproj_test_nmse_losses_ntrain_{}_ntest_{}.csv'.format(example, num, ntest), noproj_nmse, delimiter=',')
