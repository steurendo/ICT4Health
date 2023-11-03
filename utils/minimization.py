import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto


def condition_absolute_improvement(fw0, fw1, eps):
    return np.abs(fw0 - fw1) < eps


def condition_relative_improvement(fw0, fw1, eps):
    return np.abs(fw0 - fw1) / np.maximum(1, np.abs(fw0)) < eps


def condition_absolute_value(w0, w1, eps):
    return np.linalg.norm(w0 - w1) < eps


def condition_relative_value(w0, w1, eps):
    return np.linalg.norm(w0 - w1) / np.maximum(1, np.linalg.norm(w0)) < eps


def condition_max_iterations(iters, max_iters): return iters >= max_iters


class StoppingCondition(Enum):
    absolute_improvement = auto()
    relative_improvement = auto()
    absolute_value = auto()
    relative_value = auto()
    max_iterations = auto()


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
    def solve(self, stopping_condition: StoppingCondition, gamma=1e-3, cond_Nit=100, cond_eps=1e-10):
        A = self.A
        y = self.y
        w_hat0 = np.random.rand(A.shape[1])
        i = 0
        while True:
            gradient = 2 * A.T @ (A @ w_hat0 - y)
            w_hat1 = w_hat0 - gamma * gradient
            i += 1

            if stopping_condition == StoppingCondition.absolute_value:
                condition_met = condition_absolute_value(w_hat0, w_hat1, eps=cond_eps)
            elif stopping_condition == StoppingCondition.relative_value:
                condition_met = condition_relative_value(w_hat0, w_hat1, eps=cond_eps)
            elif stopping_condition == StoppingCondition.max_iterations:
                condition_met = condition_max_iterations(i, cond_Nit)
            else:
                f0 = np.linalg.norm(y - A @ w_hat0) ** 2
                f1 = np.linalg.norm(y - A @ w_hat1) ** 2
                if stopping_condition == StoppingCondition.absolute_improvement:
                    condition_met = condition_absolute_improvement(f0, f1, eps=cond_eps)
                else:
                    condition_met = condition_relative_improvement(f0, f1, eps=cond_eps)
            if condition_met:
                break
            w_hat0 = w_hat1
        self.result = w_hat1


class SolverSteepestDescent(SolverMinProblem):
    def solve(self, stopping_condition: StoppingCondition, cond_eps=1e-10, cond_Nit=100):
        A = self.A
        y = self.y
        w_hat0 = np.random.rand(A.shape[1])
        i = 0
        hessian = 2 * A.T @ A
        while True:
            gradient = 2 * A.T @ (A @ w_hat0 - y)
            N = np.linalg.norm(gradient) ** 2
            D = gradient.T @ hessian @ gradient
            gamma = N / D
            w_hat1 = w_hat0 - gamma * gradient
            i += 1

            if stopping_condition == StoppingCondition.absolute_value:
                condition_met = condition_absolute_value(w_hat0, w_hat1, eps=cond_eps)
            elif stopping_condition == StoppingCondition.relative_value:
                condition_met = condition_relative_value(w_hat0, w_hat1, eps=cond_eps)
            elif stopping_condition == StoppingCondition.max_iterations:
                condition_met = condition_max_iterations(i, cond_Nit)
            else:
                f0 = np.linalg.norm(y - A @ w_hat0) ** 2
                f1 = np.linalg.norm(y - A @ w_hat1) ** 2
                if stopping_condition == StoppingCondition.absolute_improvement:
                    condition_met = condition_absolute_improvement(f0, f1, eps=cond_eps)
                else:
                    condition_met = condition_relative_improvement(f0, f1, eps=cond_eps)
            if condition_met:
                break
            w_hat0 = w_hat1
        self.result = w_hat1
