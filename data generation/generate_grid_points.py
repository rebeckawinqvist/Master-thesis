import numpy as np
from numpy import loadtxt, savetxt
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon, LinearRing
import sys

example = str(sys.argv[1]).upper()

example_name = 'ex'+example
folder_name = 'ex'+example+'/'
filename = folder_name+example_name+'_'

points = loadtxt(filename+'cinfV.csv', delimiter = ',')
CinfH = loadtxt(filename+'cinfH.csv', delimiter = ',')
xlb = np.loadtxt(filename+'xlb.csv', delimiter=',')
xub =  np.loadtxt(filename+'xub.csv', delimiter=',')

G = CinfH[:,0:2]
h = CinfH[:,2]
numOfCons = CinfH.shape[0]
x0 = -4.5
y0 = 2

# generate grid points

x_start, x_stop = xlb[0], xub[0]
y_start, y_stop = xlb[1], xub[1]

n = 100
grid = [[x, y] for x in np.linspace(x_start,x_stop,n) for y in np.linspace(y_start,y_stop,n)]
grid = np.array(grid)

# generate feasible grid points
feasible_points = []
for point in grid:
    comp = np.less_equal(G @ point, h)
    comp = comp.tolist()
    if False not in comp: 
        feasible_points.append(point)
    
feasible_points = np.array(feasible_points)
    

linewidth = 0.8
plot = True
if plot:
    plt.axis([x_start-1,x_stop+1,y_start-1,y_stop+1])
    x = np.linspace(x_start,x_stop,1000)

    # plot constraints    
    for i in range(numOfCons):
        if CinfH[i,1] != 0:
            k = -CinfH[i,0]/CinfH[i,1]
            m = CinfH[i,2]/CinfH[i,1]
            f = k * x + m
            plt.plot(x,f, '-k', linewidth=linewidth, color='gray')
        else:
            plt.axvline(x = CinfH[i,2]/CinfH[i,0], ymin=y_start, ymax=y_stop, linewidth=linewidth, color='gray')    


    """
    # plot vertices
    for i in range(points.shape[0]):
        x1 = points[i,0]
        y1 = points[i,1]
        plt.plot(x1,y1, 'ko', linewidth=linewidth, markersize=2)
    """

    # plot initial state
    #plt.plot(x0, y0, 'o', markersize=4, color='firebrick')


    # plot feasible grid points
    for point in feasible_points:
        x, y = point[0], point[1]
        plt.plot(x, y, 'go', markersize=1)


    title = "Example {} ".format(example)
    xlabel = "$x_1$"
    ylabel = "$x_2$"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


# save
#np.savetxt(filename+'initial_states_grid.csv', feasible_points, delimiter=",")


