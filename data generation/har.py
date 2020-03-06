from scipy.stats import uniform
import numpy as np
from scipy.spatial.distance import norm
import random
import math

class HAR(object):

    def __init__(self, polytope, x0, n_samples, thin=1):
        # ensure x0 is of right dim and is inside polytope
        assert len(x0) == polytope.n
        assert polytope.is_inside(x0)

        self.polytope = polytope
        self.x0 = x0
        self.n_samples = n_samples
        self.thin = thin
        self.samples = []
        self.direction = None
        self.lamdas = []

        self.current = x0


    def sample(self, n_samples=None, thin=None):
        self.samples = []

        if n_samples is not None:
            self.n_samples = n_samples
        if thin is not None:
            self.thin = thin

        for i in range(self.n_samples):
            self.make_step()
            if i % self.thin == 0:
                self.samples.append(self.current)


    def make_step(self):
        self.get_direction()
        next_sample = self.get_random_point()

        self.current = next_sample


    def get_direction(self):
        direction = np.random.randn(self.polytope.n)
        self.direction = direction/norm(direction)


    def get_random_point(self):
        A = self.polytope.A
        b = self.polytope.b

        closest_point = None
        closest_dist = np.inf

        k,m = 0,0
        if self.direction[0] != 0:
            k = self.direction[1]/self.direction[0]
            m = self.current[1] - k*self.current[0]
        else:
            k = -1
            m = self.current[0]

        for (params, cons) in zip(A,b):
            G = np.array([[-k, 1], [params[0], params[1]]])
            h = np.array([[m], [cons]])
            x = np.linalg.solve(G,h).flatten()

            # check direction
            d = x - self.current
            d = d/norm(d)
            ang = np.arccos(round(np.dot(d, self.direction)))
            if ang == 0:
                # right direction
                dist = math.hypot(x[0]-self.current[0], x[1]-self.current[1])
                if dist < closest_dist:
                    closest_point = x
                    closest_dist = dist

        lb = min(closest_point[0], self.current[0])
        ub = max(closest_point[0], self.current[0])
        x1_new = np.random.uniform(lb, ub)
        x2_new = k*x1_new + m

        sample = np.array([x1_new, x2_new])
        return sample




