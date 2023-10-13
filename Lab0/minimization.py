import numpy as np
import matplotlib.pyplot as plt


class SolverMinProblem:
    def __init__(self, A=np.eye(3), y=np.ones(shape=(3,))):
        self.A = A
        self.y = y
        self.Np = A.shape[0]
        self.Nf = A.shape[1]
        self.result = np.zeros(shape=(self.Nf,))

    def plot_result(self):
        plt.figure()
        plt.title("LLS solution")
        plt.plot(self.result, 'w_hat')
        plt.xlabel('n')
        plt.ylabel('w_hat(n)')
        plt.grid()
        plt.show()


class SolverLLS(SolverMinProblem):
    def solve(self):
        self.result = np.linalg.pinv(self.A) @ self.y


class SolverGradientAlgorithm(SolverMinProblem):
    def solve(self):
        # TODO
        None
