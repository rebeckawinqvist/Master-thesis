from scipy.stats import uniform
import numpy as np


class Polytope(object):
    """ Polytope in H-representation 
        Ax <= b
    """

    def __init__(self, A, b, V):
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

    """
    def get_points_in_planes(self):
        self.points_in_planes = []
        for i in range(self.n_cons):
            p = np.zeros(self.n)
            j = np.argmax(self.A[i] != 0)
            p[j] = self.b[i] / self.A[i][j]
            self.points_in_planes.append(p)
    """
            

