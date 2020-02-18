import cvxpy as cp 
import torch
from cvxpylayers.torch import CvxpyLayer

m, n = 20, 10
x = cp.Variable((n, 1))
F = cp.Parameter((m, n))
g = cp.Parameter((m, 1))
lambd = cp.Parameter((1, 1), nonneg=True)
objective_fn = cp.norm(F @ x - g) + lambd * cp.norm(x)
constraints = [x >= 0]
problem = cp.Problem(cp.Minimize(objective_fn), constraints)
assert problem.is_dpp()

F_t = torch.randn(m, n, requires_grad=True)
g_t = torch.randn(m, 1, requires_grad=True)
lambd_t = torch.rand(1, 1, requires_grad=True)
print("F_t grad: ", F_t.grad)
print("g_t grad: ", g_t.grad)

# Canonicalizes problem to extract C and R
layer = CvxpyLayer(problem, parameters=[F, g, lambd], variables=[x])
# Calling layer applies the map R o s o C, returning a solution to the problem

x_star, = layer(F_t, g_t, lambd_t)

# Computes the gradients of summing x_star, with respect to F_t, g_t, lambd_t
x_star.sum().backward()
print("F_t grad: ", F_t.grad)
print("g_t grad: ", g_t.grad)