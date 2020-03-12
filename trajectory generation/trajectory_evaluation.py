import numpy as np 
import sys
import matplotlib.pyplot as plt 
from polytope import Polytope 



if __name__ == "__main__":
    example = str(sys.argv[1]).upper()
    ntrajs = int(sys.argv[2])
    N = int(sys.argv[3])
    example_name = 'ex'+example
    folder_name = 'ex'+example+'/'
    filename = folder_name+example_name+'_'

    A = np.loadtxt(filename+'A.csv', delimiter=',')
    B = np.loadtxt(filename+'B.csv', delimiter=',')
    Q = np.loadtxt(filename+'Q.csv', delimiter=',')
    R = np.loadtxt(filename+'R.csv', delimiter=',')
    m = B.ndim
    #H = np.loadtxt(filename+'cinfH.csv', delimiter=',')
    #V = np.loadtxt(filename+'cinfV.csv', delimiter=',')
    #xlb = np.loadtxt(filename+'xlb.csv', delimiter=',')
    #xub =  np.loadtxt(filename+'xub.csv', delimiter=',')
    #A_p = H[:,0:-1]
    #B_p = H[:,-1]
    #polytope = Polytope(A,B,V)
    #nsamples = 10

    to_plot = True

    names = ["Proj NN", "LQR Proj NN", "MPC"]
    costs = [[], [], []]
    costs_dict = dict(zip(names, costs))

    for i in range(ntrajs-1):
        traj_projNN = np.loadtxt(filename+'projNN_trajectory_{}_N_{}.csv'.format(i+1, N), delimiter=',')
        traj_noprojNN = np.loadtxt(filename+'noprojNN_trajectory_{}_N_{}.csv'.format(i+1, N), delimiter=',')
        traj_mpc = np.loadtxt(filename+'mpc_trajectory_{}_N_{}.csv'.format(i+1, N), delimiter=',')
        traj_lqr_projNN = np.loadtxt(filename+'lqr_projNN_trajectory_{}_N_{}.csv'.format(i+1,N), delimiter=',')

        controls_projNN = np.loadtxt(filename+'projNN_controls_trajectory_{}_N_{}.csv'.format(i+1, N), delimiter=',')
        controls_noprojNN = np.loadtxt(filename+'noprojNN_controls_trajectory_{}_N_{}.csv'.format(i+1, N), delimiter=',')
        controls_mpc = np.loadtxt(filename+'mpc_controls_trajectory_{}_N_{}.csv'.format(i+1, N), delimiter=',')
        controls_lqr_projNN = np.loadtxt(filename+'lqr_projNN_controls_trajectory_{}_N_{}.csv'.format(i+1,N), delimiter=',')

        trajs = [traj_projNN, traj_lqr_projNN, traj_mpc]
        controls = [controls_projNN, controls_lqr_projNN, controls_mpc]
        
        trajs_dict = dict(zip(names, trajs))
        controls_dict = dict(zip(names, controls))
        
        for key in trajs_dict:            
            cost = 0
            states, controls = trajs_dict[key][0:-1], controls_dict[key]
            x0 = states[0,:]
            for (x,u) in zip(states, controls):
                if m < 2:
                    c = x @ Q @ x + u * R * u
                else:
                    c = x @ Q @ x + u @ R @ u

                cost += c
            
            # normalize cost
            cost_n = cost / (x0 @ x0)
            costs_dict[key].append(cost_n)

    for key in costs_dict:
        print(key, ": ")
        for c in costs_dict[key]:
            #print(c)
            pass

        lambda_max = max(costs_dict[key])
        lambda_min = min(costs_dict[key])
        print("\nMin: {} \nMax: {} \n\n".format(lambda_min, lambda_max))       



