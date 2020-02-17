import numpy as np
from numpy import loadtxt, savetxt
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon, LinearRing

points = loadtxt('exD_cinfV.csv', delimiter = ',')
CinfH = loadtxt('exD_cinfH.csv', delimiter = ',')
G = CinfH[:,0:2]
Gx = CinfH[:,0]
Gy = CinfH[:,1]
h = CinfH[:,2]
numOfCons = CinfH.shape[0]
x0 = -4.5
y0 = 2

# generate grid points
start = -5 
stop = 5
n = 100
grid = [[x, y] for x in np.linspace(start,stop,n) for y in np.linspace(start,stop,n)]
grid = np.array(grid)

# generate feasible grid points
feasible_points = []
for point in grid:
    comp = np.less_equal(G @ point, h)
    comp = comp.tolist()
    if False not in comp: 
        feasible_points.append(point)
    
feasible_points = np.array(feasible_points)
    

linewidth = 0.4
plot = True
if plot:
    plt.axis([-5,5,-5,5])
    x = np.linspace(-5,5,1000)

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

    # plot initial state
    plt.plot(x0, y0, 'o', markersize=4, color='firebrick')


    # plot feasible grid points
    for point in feasible_points:
        x, y = point[0], point[1]
        plt.plot(x, y, 'bo', markersize=1)


    plt.show()


# save
np.savetxt("exD_initial_states_grid.csv", feasible_points, delimiter=",")


