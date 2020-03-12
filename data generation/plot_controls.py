import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from numpy import savetxt, loadtxt
import sys

example = str(sys.argv[1]).upper()
sampling_method = str(sys.argv[2].lower())

example_name = 'ex'+example
folder_name = 'ex'+example+'/'
filename = folder_name+example_name+'_'


if __name__ == "__main__":
    A = np.loadtxt(filename+'A.csv', delimiter=',')
    B = np.loadtxt(filename+'B.csv', delimiter=',')
    n, m = A.shape[1], B.ndim

    states = np.loadtxt(filename+'input_data_'+sampling_method+'.csv', delimiter=',')
    control_inputs = np.loadtxt(filename+'output_data_'+sampling_method+'.csv', delimiter=',')[:,0:m]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    i = 0
    for (x,u) in zip(states, control_inputs):
        x1, x2 = x[0], x[1]
        if i % 10 == 0:
            ax.scatter(x1, x2, u, c='b', marker='o', s=1)
        i += 1

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$u$')

    plt.show()





    
    