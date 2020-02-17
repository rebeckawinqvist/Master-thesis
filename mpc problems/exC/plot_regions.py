import numpy as np
from numpy import loadtxt, savetxt
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon, LinearRing

points = loadtxt('exC_cinfV.csv', delimiter = ',')
CinfH = loadtxt('exC_cinfH.csv', delimiter = ',')
Gx = CinfH[:,0]
Gy = CinfH[:,1]
h = CinfH[:,2]
numOfCons = CinfH.shape[0]
x0 = -4.5
y0 = 2

linewidth = 0.4
plot = True
if plot:
    plt.axis([-10,10,-10,10])
    x = np.linspace(-10,10,1000)

    # plot constraints    
    for i in range(numOfCons):
        if CinfH[i,1] != 0:
            k = -CinfH[i,0]/CinfH[i,1]
            m = CinfH[i,2]/CinfH[i,1]
            f = k * x + m
            plt.plot(x,f, '-k', linewidth=linewidth, color='gainsboro')
        else:
            plt.axvline(x = CinfH[i,2]/CinfH[i,0], ymin=-10, ymax=10, linewidth=linewidth, color='gainsboro')


    # plot vertices
    for i in range(points.shape[0]):
        x1 = points[i,0]
        y1 = points[i,1]
        plt.plot(x1,y1, 'ko', linewidth=linewidth, markersize=2)

    plt.show()
