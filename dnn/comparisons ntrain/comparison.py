import numpy as np  
import logging
logging.getLogger().setLevel(logging.INFO)
import statistics           
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from math import log10

example = str(sys.argv[1]).upper()
if len(sys.argv) > 2:
    date = sys.argv[2]
else:
    date = datetime.date(datetime.now())

filename = "ex{}/ex{}_".format(example, example)
filename_mse = "ex{}/{}/test_losses/ex{}_".format(example, date, example)
filename_tv = "ex{}/{}/true_values/ex{}_".format(example, date, example)

if example == "E":
    trains = [200, 500, 1000, 1500]
    act_trains = [135, 405, 800, 1185]

elif example == "A":
    trains = [1000]
    act_trains = trains

else:
    trains = act_trains = []

proj_mse_v, noproj_mse_v, lqr_mse_v = [], [], []
proj_nmse_v, noproj_nmse_v, lqr_nmse_v = [], [], []

f1, f2 = plt.figure(1), plt.figure(2)

ntest = 500

j = 0
for num in trains:
    proj_mse = np.loadtxt(filename_mse+'pdnn_test_mse_losses_ntrain_{}.csv'.format(num), delimiter=',')
    noproj_mse = np.loadtxt(filename_mse+'dnn_test_mse_losses_ntrain_{}.csv'.format(num), delimiter=',')
    lqr_mse = np.loadtxt(filename_mse+'lqr_pdnn_test_mse_losses_ntrain_{}.csv'.format(num), delimiter=',')

    proj_tv = np.loadtxt(filename_tv+"pdnn_true_values_ntrain_{}.csv".format(num), delimiter=',')
    noproj_tv = np.loadtxt(filename_tv+"dnn_true_values_ntrain_{}.csv".format(num), delimiter=',') 
    lqr_tv = np.loadtxt(filename_tv+"lqr_pdnn_true_values_ntrain_{}.csv".format(num), delimiter=',')

    """
    proj_nmse = np.loadtxt(filename_nmse+'test_nmse_losses_ntrain_{}_NEW.csv'.format(num), delimiter=',')
    noproj_nmse = np.loadtxt(filename_nmse+'noproj_test_nmse_losses_ntrain_{}_NEW.csv'.format(num), delimiter=',')
    lqr_nmse = np.loadtxt(filename_nmse+'lqr_test_nmse_losses_ntrain_{}_NEW.csv'.format(num), delimiter=',')
    """

    if example == "E":
        #proj_nmse = proj_nmse[1:]
        #noproj_nmse = noproj_nmse[1:]
        #lqr_nmse = lqr_nmse[1:] 

        proj_mse = proj_mse[1:]
        noproj_mse = noproj_mse[1:]
        lqr_mse = lqr_mse[1:]  

        proj_tv = proj_tv[1:]
        noproj_tv = noproj_tv[1:]
        lqr_tv = lqr_tv[1:]
 

    proj_mse_mean = np.mean(proj_mse)
    noproj_mse_mean = np.mean(noproj_mse)
    lqr_mse_mean = np.mean(lqr_mse)

    proj_tv_mean = np.mean(proj_tv)
    noproj_tv_mean = np.mean(noproj_tv)
    lqr_tv_mean = np.mean(lqr_tv)

    proj_nmse = proj_mse_mean/proj_tv_mean
    noproj_nmse = noproj_mse_mean/noproj_tv_mean
    lqr_nmse = lqr_mse_mean/lqr_tv_mean

    proj_mse_v.append(proj_mse_mean)
    proj_nmse_v.append(proj_nmse)

    lqr_mse_v.append(lqr_mse_mean)
    lqr_nmse_v.append(lqr_nmse)

    noproj_mse_v.append(noproj_mse_mean)
    noproj_nmse_v.append(noproj_nmse)
    #proj_nmse_mean = np.mean(proj_nmse)
    #noproj_nmse_mean = np.mean(noproj_nmse)
    #lqr_nmse_mean = np.mean(lqr_nmse)   

    #print(i)
    #num = act_trains[i]

    if num == trains[0]: 
        num = act_trains[j]
        plt.figure(1)
        plt.scatter(num, 10*log10(proj_mse_mean), color='r', marker='o', label='Proj NN')
        plt.scatter(num, 10*log10(lqr_mse_mean), color='g', marker='s', label='LQR')
        plt.scatter(num, 10*log10(noproj_mse_mean), color='b', marker='^', label='NoProj NN')

        plt.figure(2)
        plt.scatter(num, 10*log10(proj_nmse), color='r', marker='o', label='Proj NN')
        plt.scatter(num, 10*log10(lqr_nmse), color='g', marker='s', label='LQR')
        plt.scatter(num, 10*log10(noproj_nmse), color='b', marker='^', label='NoProj NN')

    else:
        num = act_trains[j]
        plt.figure(1)
        plt.scatter(num, 10*log10(proj_mse_mean), color='r', marker='o')
        plt.scatter(num, 10*log10(lqr_mse_mean), color='g', marker='s')
        plt.scatter(num, 10*log10(noproj_mse_mean), color='b', marker='^')

        plt.figure(2)
        plt.scatter(num, 10*log10(proj_nmse), color='r', marker='o')
        plt.scatter(num, 10*log10(lqr_nmse), color='g', marker='s')
        plt.scatter(num, 10*log10(noproj_nmse), color='b', marker='^')

    j += 1


plt.figure(1)
plt.title("Mean MSE comparison")
plt.xlabel("Number of training samples")
plt.ylabel("Mean MSE on test data")
plt.legend(loc='upper right')
plt.figure(1).show()

plt.figure(2)
plt.title("Mean NMSE comparison")
plt.xlabel("Number of training samples")
plt.ylabel("Mean NMSE on test data")
plt.legend(loc='upper right')
plt.show()


# FOR MATLAB
"""
np.savetxt("ex{}/{}/matlab_exp/ex{}_comp_act_train_samples.csv".format(example, date, example), np.array(act_trains), delimiter=',')
np.savetxt("ex{}/{}/matlab_exp/ex{}_comp_train_samples.csv".format(example, date, example), np.array(trains), delimiter=',')

np.savetxt("ex{}/{}/matlab_exp/ex{}_proj_mse_comp.csv".format(example, date, example), np.array(proj_mse_v), delimiter=',')
np.savetxt("ex{}/{}/matlab_exp/ex{}_noproj_mse_comp.csv".format(example, date, example), np.array(noproj_mse_v), delimiter=',')
np.savetxt("ex{}/{}/matlab_exp/ex{}_lqr_proj_mse_comp.csv".format(example, date, example), np.array(lqr_mse_v), delimiter=',')

np.savetxt("ex{}/{}/matlab_exp/ex{}_proj_nmse_comp.csv".format(example, date, example), np.array(proj_nmse_v), delimiter=',')
np.savetxt("ex{}/{}/matlab_exp/ex{}_noproj_nmse_comp.csv".format(example, date, example), np.array(noproj_nmse_v), delimiter=',')
np.savetxt("ex{}/{}/matlab_exp/ex{}_lqr_proj_nmse_comp.csv".format(example, date, example), np.array(lqr_nmse_v), delimiter=',')

"""