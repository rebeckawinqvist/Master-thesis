from __future__ import print_function
import sys

import numpy as np 
import matplotlib.pyplot as plt 
import polytope

if __name__ == "__main__":
    if len(sys.argv) < 2:
        N = 3
    else:
        N = int(sys.argv[1])

    V = np.random.randn(N,2)

    print("Sampled " + str(N)+ " points:")
    print(V)

    P = polytope.qhull(V)
    print("Computed the convex hull:")
    print(P)

    V_min = polytope.extreme(P)
    print("Which has extreme points:")
    print(V_min)

    P.plot()
    plt.show()