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

h = CinfH[:,2]
numOfCons = CinfH.shape[0]
x0 = -4.5
y0 = 2

x_start, x_stop = xlb[0], xub[0]
y_start, y_stop = xlb[1], xub[1]

linewidth = 0.8
plot = True
if plot:
    plt.axis([x_start-1,x_stop-1,y_start-1,y_stop+1])
    x = np.linspace(x_start,x_stop,1000)

    # plot constraints    
    for i in range(numOfCons):
        if CinfH[i,1] != 0:
            k = -CinfH[i,0]/CinfH[i,1]
            m = CinfH[i,2]/CinfH[i,1]
            f = k * x + m
            plt.plot(x,f, '-k', linewidth=linewidth, color='gainsboro')
        else:
            plt.axvline(x = CinfH[i,2]/CinfH[i,0], ymin=y_start, ymax=y_stop, linewidth=linewidth, color='gainsboro')

    # plot vertices
    for i in range(points.shape[0]):
        x1 = points[i,0]
        y1 = points[i,1]
        plt.plot(x1,y1, 'ko', linewidth=linewidth, markersize=2)

    # plot initial state
    #plt.plot(x0, y0, 'o', markersize=4, color='firebrick')


    # plot lines between vertices and initial state
    lines = []
    for i in range(points.shape[0]):
        xV = points[i,0]
        yV = points[i,1]
        x = [x0, xV]
        y = [y0, yV]
        if x0 != xV:
            k = (y0-yV)/(x0-xV)
            m = y0 - k*x0
        else:
            continue
        params = [k, m, x0, xV, y0, yV]
        lines.append(params)
        plt.plot(x,y, linewidth=linewidth, color='green')

    # plot additional points on lines
    addPoints = []
    xArray = []
    yArray = []
    for param in lines:
        k, m, x0, xV, y0, yV = param[0], param[1], param[2], param[3], param[4], param[5]
        if x0 > xV:
            xRange = np.linspace(x_start, x0, 10)
        else:
            xRange = np.linspace(x0, xV, 10)

        for x in xRange:
            y = k *x + m
            point = (x,y)
            addPoints.append(point)
            xArray.append(x)
            yArray.append(y)

    
    # plot points
    for p in addPoints:
        plt.plot(p[0], p[1], 'go', markersize=3)

    
    title = "Example {} ".format(example)
    xlabel = "$x_1$"
    ylabel = "$x_2$"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()

print("Added points: ", len(addPoints))

# convert lists into arrays
xArray = np.array(xArray)
yArray = np.array(yArray)
points = np.column_stack((xArray, yArray))

np.savetxt(filename+'initial_states_rays.csv', points, delimiter=",")


