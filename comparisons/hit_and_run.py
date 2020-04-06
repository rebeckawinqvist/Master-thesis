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
    gen = str(sys.argv[2])
    nsamples = int(sys.argv[3])

    gen_type = ''
    if gen.upper() == "TRAIN" or gen.upper() == "TEST":
        gen_type = gen.lower()
    else:
        raise ValueError


    example_name = 'ex'+example
    folder_name = 'ex'+example+'/'
    filename = folder_name+example_name+'_'

    H = np.loadtxt(filename+'cinfH.csv', delimiter=',')
    #V = np.loadtxt(filename+'cinfV.csv', delimiter=',')
    xlb = np.loadtxt(filename+'xlb.csv', delimiter=',')
    xub =  np.loadtxt(filename+'xub.csv', delimiter=',')
    A = H[:,0:-1]
    B = H[:,-1]
    polytope = Polytope(A,B)
    x0 = np.array([-4.5, 2])
    if polytope.n == 4:
        x0 = np.array([0, 0, 0, 0])
        print(polytope.is_inside(x0))

    to_plot = True
    if polytope.n > 2:
        to_plot = False 


    #plt.arrow(x0[0], x0[1], direction[0], direction[1], head_width=0.1, head_length=0.05)

    samples = [x0]
    for n in range(nsamples-1):
        direction = np.random.randn(polytope.n)
        direction = direction/norm(direction)
        
        smallest_lambda = np.inf
        # system of equations
        for (a, b) in zip(A,B):
            l = (b - np.dot(a,x0)) / (np.dot(a,direction))
            x_l = x0 + l*direction
            if l < 0:
                continue
            
            if l < smallest_lambda and l != 0:
                smallest_lambda = l    

        # Generate random point
        l = np.random.uniform(0, smallest_lambda)
        sample = x0 + l*direction
        samples.append(sample)
        x0 = sample


    feasible_samples = []
    for sample in samples: 
        if not polytope.is_inside(sample):
            print("Infeasible")
        else:
            feasible_samples.append(sample)
    
    if to_plot:
        plt.ylim(xlb[1]-1,xub[1]+1)
        plt.xlim(xlb[0]-1,xub[0]+1)
        polytope.plot_poly(xlb, xub, save=False, show=False)

        for sample in feasible_samples: 
            plt.plot(sample[0], sample[1], 'go', linewidth=0.8, markersize=2)

        if gen_type == 'train':
            gen_title = "Training"
        else:
            gen_title = "Test"

        title = "Example {} \n (HAR) {} samples: {}".format(example, gen_title, len(feasible_samples))
        xlabel = "$x_1$"
        ylabel = "$x_2$"
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel) 
        filen_fig = filename+"har_nsamples_{}_{}.png".format(len(feasible_samples), gen_type)

        plt.savefig(filen_fig)

        plt.show()


    print("Feasible samples [%]: ", 100*len(feasible_samples)/len(samples))
    print("Kept samples: ", len(feasible_samples),"/",len(samples))

    np.savetxt("ex{}/har_samples/ex{}_har_nsamples_{}_{}.csv".format(example, example, len(feasible_samples), gen_type), feasible_samples, delimiter=",")

    



