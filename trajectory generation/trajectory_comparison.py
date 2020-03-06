import numpy as np 
import sys
import matplotlib.pyplot as plt 
from polytope import Polytope 



if __name__ == "__main__":
    example = str(sys.argv[1]).upper()
    ntrajs = int(sys.argv[2])
    N = int(sys.argv[3])
    example_name = 'ex'+example
    filename = example_name+'_'

    H = np.loadtxt(filename+'cinfH.csv', delimiter=',')
    V = np.loadtxt(filename+'cinfV.csv', delimiter=',')
    xlb = np.loadtxt(filename+'xlb.csv', delimiter=',')
    xub =  np.loadtxt(filename+'xub.csv', delimiter=',')
    A = H[:,0:-1]
    B = H[:,-1]
    polytope = Polytope(A,B,V)
    nsamples = 10

    to_plot = True

    for i in range(nsamples):
        traj_proj_NN = np.loadtxt(filename+'projNN_trajectory_{}_N_{}.csv'.format(i+1, N), delimiter=',')
        traj_noproj_NN = np.loadtxt(filename+'noprojNN_trajectory_{}_N_{}.csv'.format(i+1, N), delimiter=',')
        traj_mpc = np.loadtxt(filename+'mpc_trajectory_{}_N_{}.csv'.format(i+1, N), delimiter=',')

        #trajs = [traj_proj_NN, traj_noproj_NN, traj_mpc]
        #names = ["Proj NN", "NoProj NN", "MPC"]
        #table = dict(zip(names, trajs))


        if to_plot:
            # Points
            plt.subplot(1,2,1)
            polytope.plot_poly(xlb, xub, show=False)
            plt.scatter(traj_noproj_NN[:,0], traj_noproj_NN[:,1], color='r', label="NoProj NN")
            plt.scatter(traj_proj_NN[:,0], traj_proj_NN[:,1], color='b', label="Proj NN")
            plt.scatter(traj_mpc[:,0], traj_mpc[:,1], color='g', label="MPC")
            plt.legend()

            title = "Example {} \n Sample: {} \n ".format(example,i+1)
            xlabel = "$x_1$"
            ylabel = "$x_2$"
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel) 

            #filen_fig = filename+'traj_comp_sample_{}_nsim_{}_points.png'.format(i+1, nsim)
            #plt.savefig(filen_fig)

            #plt.show()

            
            # Lines
            plt.subplot(1,2,2)
            polytope.plot_poly(xlb, xub, show=False)
            plt.plot(traj_noproj_NN[:,0], traj_noproj_NN[:,1], color='r', label="NoProj NN")
            plt.plot(traj_proj_NN[:,0], traj_proj_NN[:,1], color='b', label="Proj NN")
            plt.plot(traj_mpc[:,0], traj_mpc[:,1], color='g', label="MPC")
            plt.legend()

            title = "Example {} \n Sample: {} \n ".format(example,i+1)
            xlabel = "$x_1$"
            ylabel = "$x_2$"
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel) 

            #filen_fig = filename+'traj_comp_sample_{}_nsim_{}_lines.png'.format(i+1, nsim)
            #plt.savefig(filen_fig)


            filen_fig = filename+'comparison_trajectory_1_{}_N_{}.png'.format(i+1, N)
            plt.savefig(filen_fig)
            plt.tight_layout()
            plt.subplots_adjust(top=0.832, bottom=0.132, left=0.07, right=0.979, hspace=0.2, wspace=0.179)

            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()

            plt.show()
        



