from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
from har import * 
from polytope import *
import sys

example = str(sys.argv[1]).upper()

example_name = 'ex'+example
filename = example_name+'_'

def plot(H, x0, samples):
    plt.axis([-6,6,-5,5])
    x = np.linspace(-6,6,1000)

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

    plt.show()


if __name__ == "__main__":
    H = np.loadtxt(filename+'cinfH.csv', delimiter=',')
    V = np.loadtxt(filename+'cinfV.csv', delimiter=',')
    A = H[:,0:-1]
    b = H[:,-1]
    x0 = np.array([-4.5, 2])
    polytope = Polytope(A, b, V)
    if polytope.n == 4:
        x0 = np.array([-2, -5.3, 1, 0.5])
    n_samples = 1000

    har = HAR(polytope, x0, n_samples)

    har.get_samples()
    np.savetxt(filename+'initial_states_har.csv', har.samples, delimiter=",")

    #plot(H, x0, har.samples)









