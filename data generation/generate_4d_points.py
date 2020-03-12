import numpy as np
from numpy import loadtxt, savetxt
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon, LinearRing
import sys

example = str(sys.argv[1]).upper()

example_name = 'ex'+example
folder_name = 'ex'+example+'/'
filename = folder_name+example_name+'_'

CinfH = loadtxt(filename+'cinfH.csv', delimiter = ',')
xlb = np.loadtxt(filename+'xlb.csv', delimiter=',')
xub =  np.loadtxt(filename+'xub.csv', delimiter=',')

G = CinfH[:,0:4]
h = CinfH[:,4]
numOfCons = CinfH.shape[0]

x1_start, x2_start, x3_start, x4_start = xlb[0], xlb[1], xlb[2], xlb[3]
x1_stop, x2_stop, x3_stop, x4_stop = xub[0], xub[1], xub[2], xub[3]

n = 20
grid = [[x1, x2, x3, x4] for x1 in np.linspace(x1_start, x1_stop, n) 
                         for x2 in np.linspace(x2_start, x2_stop, n)
                         for x3 in np.linspace(x3_start, x3_stop, n)
                         for x4 in np.linspace(x4_start, x4_stop, n)]
grid = np.array(grid)


# generate feasible points
feasible_points = []
for point in grid:
    comp = np.less_equal(G @ point, h)
    comp = comp.tolist()
    if False not in comp: 
        feasible_points.append(point)
    
feasible_points = np.array(feasible_points)
   
np.savetxt(filename+'initial_states_grid.csv', feasible_points, delimiter=",")

