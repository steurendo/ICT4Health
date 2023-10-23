import numpy as np
import matplotlib.pyplot as plt


def condition_absolute_improvement(fw0, fw1, eps):
    return np.abs(fw0 - fw1) < eps


def condition_relative_improvement(fw0, fw1, eps):
    return np.abs(fw0 - fw1) / np.maximum(1, np.abs(fw0)) < eps


def condition_absolute_value(w0, w1, eps):
    return np.linalg.norm(w0 - w1) < eps


def condition_relative_value(w0, w1, eps):
    return np.linalg.norm(w0 - w1) / np.maximum(1, np.linalg.norm(w0)) < eps


class SolverMinProblem:
    def __init__(self, A=np.eye(3), y=np.ones(shape=(3,))):
        self.A = A
        self.y = y
        self.Np = A.shape[0]
        self.Nf = A.shape[1]
        self.result = np.zeros(shape=(self.Nf,))

    def plot_result(self):
        plt.figure()
        plt.title('LLS solution')
        plt.plot(self.result, 'w_hat')
        plt.xlabel('n')
        plt.ylabel('w_hat(n)')
        plt.grid()
        plt.show()


class SolverLLS(SolverMinProblem):
    def solve(self):
        A = self.A
        y = self.y
        self.result = np.linalg.pinv(A) @ y


class SolverGradient(SolverMinProblem):
    def solve(self, gamma=1e-3, Nit=100):
        A = self.A
        y = self.y
        w_hat0 = np.random.rand(A.shape[1])
        i = 0
        while True:
            gradient = 2 * A.T @ (A @ w_hat0 - y)
            w_hat1 = w_hat0 - gamma * gradient
            i += 1

            err0 = np.linalg.norm(y - A @ w_hat0) ** 2
            err1 = np.linalg.norm(y - A @ w_hat1) ** 2
            # print('Error at ', i, ' iteration: ', err1)
            if i >= Nit > 0 or condition_relative_improvement(err0, err1, eps=1e-30):
                break
            w_hat0 = w_hat1
        self.result = w_hat1


class SolverSteepestDescent(SolverLLS):
    def solve(self, Nit=100):
        A = self.A
        y = self.y
        w_hat0 = np.random.rand(A.shape[1])
        i = 0
        print("culooooooo")
        hessian = 2 * A.T @ A
        while True:
            gradient = 2 * A.T @ (A @ w_hat0 - y)
            N = np.linalg.norm(gradient) ** 2
            D = gradient.T @ hessian @ gradient
            gamma = N / D
            w_hat1 = w_hat0 - gamma * gradient
            i += 1

            err0 = np.linalg.norm(y - A @ w_hat0) ** 2
            err1 = np.linalg.norm(y - A @ w_hat1) ** 2
            # print('Error at ', i, ' iteration: ', err1)
            if i >= Nit > 0 or condition_relative_improvement(err0, err1, eps=1e-30):
                break
            w_hat0 = w_hat1
        self.result = w_hat1
