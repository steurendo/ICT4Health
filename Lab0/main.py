import numpy as np
from minimization import SolverLLS

np.random.seed(72)

Np = 5
Nf = Np - 1
A = np.random.randn(Np, Nf)
w = np.random.randn(Nf)     # here we emulate w just to check that the algo works
y = A @ w

solver = SolverLLS(A, y)
solver.solve()

w_hat = solver.result
e = y - A @ w_hat

print('LLS method results')
print('Estimated w_hat: ', w_hat)
print('True vector: ', w)
print('Square error ||y - A*w_hat|| = ', np.linalg.norm(e, 2))
