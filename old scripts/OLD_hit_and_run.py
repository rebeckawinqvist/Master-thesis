from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
from har import * 
from polytope import *
import sys

example = str(sys.argv[1]).upper()

example_name = 'ex'+example
filename = example_name+'_'

def plot(H, x0, samples, xlb, xub, ylb, yub):
    plt.axis([xlb-1, xub+1, ylb-1, yub+1])
    x = np.linspace(-11,11,1000)

    for i in range(H.shape[0]):
        if H[i,1] != 0:
            k = -H[i,0]/H[i,1]
            m = H[i,2]/H[i,1]
            f = k * x + m
            plt.plot(x,f, '-k', linewidth=0.5, color='gainsboro')
        else:
            plt.axvline(x = H[i,2]/H[i,0], ymin=-10, ymax=10, linewidth=0.5, color='gainsboro')

    plt.plot(x0[0], x0[1], 'ro', linewidth=0.8, markersize=2)

    for sample in samples: 
        plt.plot(sample[0], sample[1], 'go', linewidth=0.8, markersize=2)


    #plt.quiver(x0[0], x0[1], direction[0], direction[1], width=1/300)
        
    title = "Example {} \n HAR {} samples".format(example, len(samples))
    xlabel = "$x_1$"
    ylabel = "$x_2$"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 
    filen_fig = filename+"har"+("_samples_{}".format(len(samples)))+".png"

    plt.savefig(filen_fig)
    plt.show()


if __name__ == "__main__":
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

    har = HAR(polytope, x0, n_samples)

    har.sample()
    np.savetxt(filename+"initial_states_har.csv", har.samples, delimiter=",")

    x_start, x_stop = xlb[0], xub[0]
    y_start, y_stop = xlb[1], xub[1]
    if polytope.n <= 2:
        plot(H, x0, har.samples, x_start, x_stop, y_start, y_stop)









