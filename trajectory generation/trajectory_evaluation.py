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
    filename_trajs = example_name+'/trajectories/'
    filename_evals = example_name+'/evaluations/'

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

    names = ["Proj NN", "LQR Proj NN", "MPC", "NoProj NN"]
    costs = [[], [], [], []]
    costs_dict = dict(zip(names, costs))

    for i in range(ntrajs):
        traj_projNN = np.loadtxt(filename_trajs+'projNN_ntrajs_{}_N_{}_traj_{}.csv'.format(ntrajs, N, i+1), delimiter=',')
        traj_noprojNN = np.loadtxt(filename_trajs+'noprojNN_ntrajs_{}_N_{}_traj_{}.csv'.format(ntrajs, N, i+1), delimiter=',')
        traj_mpc = np.loadtxt(filename_trajs+'mpc_ntrajs_{}_N_{}_traj_{}.csv'.format(ntrajs, N, i+1), delimiter=',')
        traj_lqr_projNN = np.loadtxt(filename_trajs+'lqr_projNN_ntrajs_{}_N_{}_traj_{}.csv'.format(ntrajs, N, i+1), delimiter=',')

        controls_projNN = np.loadtxt(filename_trajs+'projNN_controls_ntrajs_{}_N_{}_traj_{}.csv'.format(ntrajs, N, i+1), delimiter=',')
        controls_noprojNN = np.loadtxt(filename_trajs+'noprojNN_controls_ntrajs_{}_N_{}_traj_{}.csv'.format(ntrajs, N, i+1), delimiter=',')
        controls_mpc = np.loadtxt(filename_trajs+'mpc_controls_ntrajs_{}_N_{}_traj_{}.csv'.format(ntrajs, N, i+1), delimiter=',')
        controls_lqr_projNN = np.loadtxt(filename_trajs+'lqr_projNN_controls_ntrajs_{}_N_{}_traj_{}.csv'.format(ntrajs, N, i+1), delimiter=',')

        trajs = [traj_projNN, traj_lqr_projNN, traj_mpc, traj_noprojNN]
        controls = [controls_projNN, controls_lqr_projNN, controls_mpc, controls_noprojNN]
        colors = ['b', 'g', 'y', 'r']
        
        trajs_dict = dict(zip(names, trajs))
        controls_dict = dict(zip(names, controls))
        color_dict = dict(zip(names, colors))
        
        for key in trajs_dict:            
            cost = 0
            states, controls, color = trajs_dict[key][0:-1], np.expand_dims(controls_dict[key], axis=0), color_dict[key]
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
        print(key + ": ")
        for c in costs_dict[key]:
            #print(c)
            pass
        
        lambda_max = max(costs_dict[key])
        lambda_min = min(costs_dict[key])
        print("\nMin: {} \nMax: {} \n\n\n".format(lambda_min, lambda_max))   

    
    # plot and save costs

    ms = 2
    lw = 1

    x = [i+1 for i in range(ntrajs)]
    plt.plot(x, costs_dict["Proj NN"], color='b', linestyle='--', marker='o', label='Proj NN', markersize=ms, linewidth=lw)
    plt.plot(x, costs_dict["NoProj NN"], color='r', linestyle='-.', marker='^', label='NoProj NN', markersize=ms, linewidth=lw)
    plt.plot(x, costs_dict["LQR Proj NN"], color='g', linestyle=':', marker='s', label='LQR NN', markersize=ms, linewidth=lw)
    plt.plot(x, costs_dict["MPC"], color='y', linestyle='-', marker='*', label='MPC', markersize=ms, linewidth=lw)
    plt.legend(loc = 'upper left')

    plt.title("Trajectory evaluation")
    plt.xlabel("Trajectory")
    plt.ylabel("Cost")

    plt.xticks(np.arange(0, ntrajs+1, step=1))
    filen_fig = filename_evals+'evaluation_ntrajs_{}_N_{}_alltrajs.png'.format(ntrajs, N)
    plt.savefig(filen_fig)
    plt.show()

    plt.scatter(x, costs_dict["Proj NN"], color='b', linestyle='--', marker='o', label='Proj NN', s=ms)
    plt.scatter(x, costs_dict["NoProj NN"], color='r', linestyle='-.', marker='^', label='NoProj NN', s=ms)
    plt.scatter(x, costs_dict["LQR Proj NN"], color='g', linestyle=':', marker='s', label='LQR NN', s=ms)
    plt.scatter(x, costs_dict["MPC"], color='y', linestyle='-', marker='*', label='MPC', s=ms)
    plt.legend(loc = 'upper left')
    plt.show()

    np.savetxt(filename_evals+'eval_projNN_ntrajs_{}_N_{}_alltrajs'.format(ntrajs, N)+".csv", costs_dict["Proj NN"], delimiter=',')
    np.savetxt(filename_evals+'eval_noprojNN_ntrajs_{}_N_{}_alltrajs'.format(ntrajs, N)+".csv", costs_dict["NoProj NN"], delimiter=',')
    np.savetxt(filename_evals+'eval_mpc_ntrajs_{}_N_{}_alltrajs'.format(ntrajs, N)+".csv", costs_dict["MPC"] , delimiter=',')
    np.savetxt(filename_evals+'eval_lqr_projNN_ntrajs_{}_N_{}_alltrajs'.format(ntrajs, N)+".csv", costs_dict["LQR Proj NN"] , delimiter=',')