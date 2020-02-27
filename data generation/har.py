from scipy.stats import uniform
import numpy as np
from scipy.spatial.distance import norm
import random

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
        l = self.get_distance()

        next = self.current + l * self.direction
        self.current = next


    def get_direction(self):
        direction = np.random.randn(self.polytope.n)
        self.direction = direction/norm(direction)


    def get_distance(self):
        self.set_lambdas()
        l = random.choice(self.lambdas)
        return l

    
    def set_lambdas(self):
        self.lambdas = []
        lambda_min = 0
        lambda_max = 10

        lambda_range = np.linspace(lambda_min, lambda_max, 1000)
        for i in range(len(lambda_range)-1):
            l = lambda_range[i]
            next_point = self.current + l * self.direction
            # check constraints
            if self.polytope.is_inside(next_point):
                self.lambdas.append(l)






