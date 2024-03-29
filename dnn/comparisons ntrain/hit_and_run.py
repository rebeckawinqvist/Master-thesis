from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import norm
import random
import sys
from polytope import *
import math


if __name__ == "__main__":
    example = str(sys.argv[1]).upper()
    nsamples = int(sys.argv[2])

    example_name = 'ex'+example
    folder_name = 'ex'+example+'/'
    filename = folder_name+example_name+'_'

    H = np.loadtxt(filename+'cinfH.csv', delimiter=',')
    V = np.loadtxt(filename+'cinfV.csv', delimiter=',')
    xlb = np.loadtxt(filename+'xlb.csv', delimiter=',')
    xub =  np.loadtxt(filename+'xub.csv', delimiter=',')
    A = H[:,0:-1]
    B = H[:,-1]
    polytope = Polytope(A,B,V)

    x0 = np.array([-4.5, 2])
    samples = [x0]
    n_samples = nsamples

    if polytope.n == 4:
        x0 = np.array([0, 0, 0, 0])
        samples = []
        n_samples = nsamples+1

    plot_and_save = False
    if polytope.n <= 2:
        plot_and_save = True

    #n_samples = 10


    #plt.arrow(x0[0], x0[1], direction[0], direction[1], head_width=0.1, head_length=0.05)

    for n in range(n_samples-1):
        direction = np.random.randn(polytope.n)
        direction = direction/norm(direction)
        
        smallest_lambda = np.inf
        # system of equations
        for (a, b) in zip(A,B):
            l = (b - np.dot(a,x0)) / (np.dot(a,direction))
            if l < 0:
                continue
            
            if l < smallest_lambda and l != 0:
                smallest_lambda = l
    

        # Generate random point
        l = np.random.uniform(0, smallest_lambda)
        sample = x0 + l*direction
        samples.append(sample)
        x0 = sample

    
    if plot_and_save:
        plt.ylim(xlb[1]-1,xub[1]+1)
        plt.xlim(xlb[0]-1,xub[0]+1)
        polytope.plot_poly(xlb, xub, save=False, show=False)

        for sample in samples: 
            plt.plot(sample[0], sample[1], 'go', linewidth=0.8, markersize=2)

        title = "Example {} \n HAR {} samples".format(example, len(samples))
        xlabel = "$x_1$"
        ylabel = "$x_2$"
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel) 
        filen_fig = "ex{}/initial_states/ex{}_initial_states_{}.png".format(example, example, nsamples)
        #filen_fig = filename+"har"+("_samples_{}".format(len(samples)))+".png"

        plt.savefig(filen_fig)

        plt.show()


    feasible_samples = []
    for sample in samples: 
        if polytope.is_inside(sample):
            feasible_samples.append(sample)

    print("Feasible samples: {} % ".format(100*len(feasible_samples)/len(samples))) 
    print("Kept samples: ", len(feasible_samples),"/",len(samples))

    np.savetxt("ex{}/initial_states/ex{}_initial_states_{}.csv".format(example, example, nsamples), feasible_samples, delimiter=",")
    



