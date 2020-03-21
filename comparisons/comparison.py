import numpy as np  
import logging
logging.getLogger().setLevel(logging.INFO)
import statistics           
import sys
import matplotlib.pyplot as plt

example = str(sys.argv[1]).upper()

example_name = 'ex'+example
#folder_name_mse, folder_name_nmse = 'ex'+example+'/'+
filename_mse, filename_nmse = 'ex'+example+'/mse/ex'+example+'_', 'ex'+example+'/nmse/ex'+example+'_'

start, stop, step = 100, 900, 100
f1, f2 = plt.figure(1), plt.figure(2)

for i in range(start, stop+1, step):
        proj_mse = np.loadtxt(filename_mse+'test_mse_losses_ntrain_{}.csv'.format(i), delimiter=',')
        noproj_mse = np.loadtxt(filename_mse+'noproj_test_mse_losses_ntrain_{}.csv'.format(i), delimiter=',')
        lqr_mse = np.loadtxt(filename_mse+'lqr_test_mse_losses_ntrain_{}.csv'.format(i), delimiter=',')

        proj_nmse = np.loadtxt(filename_nmse+'test_nmse_losses_ntrain_{}.csv'.format(i), delimiter=',')
        noproj_nmse = np.loadtxt(filename_nmse+'noproj_test_nmse_losses_ntrain_{}.csv'.format(i), delimiter=',')
        lqr_nmse = np.loadtxt(filename_nmse+'lqr_test_nmse_losses_ntrain_{}.csv'.format(i), delimiter=',')


        proj_mse_mean = np.mean(proj_mse)
        noproj_mse_mean = np.mean(noproj_mse)
        lqr_mse_mean = np.mean(lqr_mse)

        proj_nmse_mean = np.mean(proj_nmse)
        noproj_nmse_mean = np.mean(noproj_nmse)
        lqr_nmse_mean = np.mean(lqr_nmse)


        if i == start: 
            plt.figure(1)
            plt.scatter(i, proj_mse_mean, color='r', marker='o', label='Proj NN')
            plt.scatter(i, lqr_mse_mean, color='g', marker='s', label='LQR')
            plt.scatter(i, noproj_mse_mean, color='b', marker='^', label='NoProj NN')

            plt.figure(2)
            plt.scatter(i, proj_nmse_mean, color='r', marker='o', label='Proj NN')
            plt.scatter(i, lqr_nmse_mean, color='g', marker='s', label='LQR')
            plt.scatter(i, noproj_nmse_mean, color='b', marker='^', label='NoProj NN')

        else:
            plt.figure(1)
            plt.scatter(i, proj_mse_mean, color='r', marker='o')
            plt.scatter(i, lqr_mse_mean, color='g', marker='s')
            plt.scatter(i, noproj_mse_mean, color='b', marker='^')

            plt.figure(2)
            plt.scatter(i, proj_nmse_mean, color='r', marker='o')
            plt.scatter(i, lqr_nmse_mean, color='g', marker='s')
            plt.scatter(i, noproj_nmse_mean, color='b', marker='^')


plt.figure(1)
plt.title("Mean MSE comparison")
plt.xlabel("Number of training samples")
plt.ylabel("Mean MSE on test data")
plt.legend(loc='upper right')
#plt.figure(1).show()

plt.figure(2)
plt.title("Mean NMSE comparison")
plt.xlabel("Number of training samples")
plt.ylabel("Mean NMSE on test data")
plt.legend(loc='upper right')
plt.show()


