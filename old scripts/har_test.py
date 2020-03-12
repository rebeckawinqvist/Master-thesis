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

    example_name = 'ex'+example
    filename = example_name+'_'

    H = np.loadtxt(filename+'cinfH.csv', delimiter=',')
    V = np.loadtxt(filename+'cinfV.csv', delimiter=',')
    xlb = np.loadtxt(filename+'xlb.csv', delimiter=',')
    xub =  np.loadtxt(filename+'xub.csv', delimiter=',')
    A = H[:,0:-1]
    b = H[:,-1]
    x0 = np.array([-4.5, 2])
    polytope = Polytope(A, b, V)
    if polytope.n == 4:
        x0 = np.array([-2, -5.3, 1, 0.5])
    n_samples = 10000

    polytope.plot_poly(xlb, xub, show=False, color='gainsboro')
    


    #plt.arrow(x0[0], x0[1], direction[0], direction[1], head_width=0.1, head_length=0.05)
    plt.ylim(xlb[1]-1,xub[1]+1)
    plt.xlim(xlb[0]-1,xub[0]+1)
    plt.plot(x0[0], x0[1], 'ro', markersize=4)
 

    samples = [x0]
    for n in range(n_samples):
        closest_point = None
        closest_dist = np.inf

        direction = np.random.randn(polytope.n)
        direction = direction/norm(direction)

        k, m = 0,0
        if direction[1] != 0:
            k = direction[1]/direction[0]
            m = x0[1] - k*x0[0]
        else:
            k = -1
            m = x0[0]

        # system of equations
        for (params, cons) in zip(A,b):
            G = np.array([[-k, 1], [params[0], params[1]]])
            h = np.array([[m], [cons]])
            x = np.linalg.solve(G,h).flatten()
            #plt.plot(x[0], x[1], 'ko', markersize=2)

            # check direction
            d = x - x0
            d = d/norm(d)
            ang = np.arccos(round(np.dot(d, direction)))
            if ang == 0:
                # right direction
                dist = math.hypot(x[0] - x0[0], x[1] - x0[1])
                if dist < closest_dist:
                    closest_point = x
                    closest_dist = dist

        #x_v = np.linspace(xlb[0], xub[0], 1000)
        #f = k*x_v + m
        #plt.plot(x_v, f, linewidth=0.7)

        # closest point
        #plt.plot(closest_point[0], closest_point[1], 'bo', markersize=3)

        #d = closest_point - x0
        #d = d/norm(d)
        #plt.arrow(x0[0], x0[1], d[0], d[1], head_width=0.1, head_length=0.05, linestyle='-')
        
        #print(ang)
        #if ang != 0:
            #print("Wrong direction")
        #else:
            #print("Right direction")
        

        # Generate random point
        lb = min(closest_point[0], x0[0])
        ub = max(closest_point[0], x0[0])
        x1_new = np.random.uniform(lb, ub)
        x2_new = k*x1_new + m

        sample = np.array([x1_new, x2_new])
        samples.append(sample)

        #plt.plot(x1_new, x2_new, 'go', markersize=4)
        x0 = sample

    

    for sample in samples:
        plt.plot(sample[0], sample[1], 'go', markersize=1)
    #plt.quiver(x0[0], x0[1], direction[0], direction[1], scale=30)
    plt.show()


