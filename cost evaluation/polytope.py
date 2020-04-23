from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt


class Polytope(object):
    """ Polytope in H-representation 
        Ax <= b
    """

    def __init__(self, A, b, V=None):
        # check dimensions
        assert A.shape[0] == len(b)
        self.A = A
        self.b = b
        self.n = A.shape[1]
        self.n_cons = A.shape[0]
        self.vertices = V
        #self.get_points_in_planes()
    
    def is_inside(self, point):
        cons_check = self.A @ point <= self.b
        inside = np.all(cons_check)
        return inside

    def plot_poly(self, xlbs, xubs, infeasible_states = [], save=False, show=True, color='gray'):
        # plot constraints
        x_start, x_stop = xlbs[0], xubs[0]
        y_start, y_stop = xlbs[1], xubs[1]

        plt.axis([x_start-1,x_stop+1,y_start-1,y_stop+1])
        x = np.linspace(x_start,x_stop,1000)

        linewidth=0.8

        for i in range(self.n_cons):
            if self.A[i,1] != 0:
                k = -self.A[i,0]/self.A[i,1]
                m = self.b[i]/self.A[i,1]
                f = k * x + m
                plt.plot(x,f, '-k', linewidth=linewidth, color=color)
            else:
                plt.axvline(x = self.b[i]/self.A[i,0], ymin=y_start, ymax=y_stop, linewidth=linewidth, color='gray')

        if infeasible_states:
            for state in infeasible_states:
                x1 = state[0]
                x2 = state[1]
                plt.plot(x1,x2, 'ro', linewidth=linewidth, markersize=4)   

        if show:
            plt.show()




    """
    def get_points_in_planes(self):
        self.points_in_planes = []
        for i in range(self.n_cons):
            p = np.zeros(self.n)
            j = np.argmax(self.A[i] != 0)
            p[j] = self.b[i] / self.A[i][j]
            self.points_in_planes.append(p)
    """
            

