import pickle
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SVM:

    def __init__(self, max_iteration=1000, kernel_type='linear', regularization=1.0, learning_rate=0.001, tol=1e-5):
        self.max_iteration = max_iteration
        if kernel_type == 'linear':
            self.kernel = self.kernel_linear
        elif kernel_type == 'poly':
            self.kernel = self.kernel_poly
        elif kernel_type == 'rbf':
            self.kernel = self.kernel_rbf
        else:
            print('Wrong kernel name')

        self.kernel_type = kernel_type
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.tol = tol
        self.alphas = None
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        np.random.seed(1234)

    def kernel_linear(self, x1, x2):
        return x1 @ x2.T

    def kernel_poly(self, x1, x2, degree=2):
        if degree == 1:
            self.kernel_linear(x1, x2)
        elif degree > 1:
            res = x1 @ x2.T
            bias = 1
            for i in range(len(res)):
                res[i, :] += bias
                res[i, :] = res[i, :] ** degree
            return res
        else:
            print('Incorrect degree')

    def kernel_rbf(self, x1, x2, sigma=1):
        return np.exp(- (np.linalg.norm(x1 - x2, 2, axis=-1) ** 2) / (2 * sigma ** 2)).reshape(-1, 1)

    def train(self, X, y):
        self.X = np.mat(SVM.normalize(X))
        n = len(X)
        self.y = np.mat(y).T
        self.b = 0.0
        self.alphas = np.mat(np.zeros((n, 1)))
        passes = 0
        max_passes = int(self.max_iteration / 10)

        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(n):
                f_xi = float(np.multiply(self.alphas, self.y).T * self.kernel(self.X, self.X[i, :])) + self.b
                E_i = f_xi - float(y[i])

                if (self.y[i] * E_i < -self.tol and self.alphas[i] < self.regularization) or (self.y[i] * E_i > self.tol and self.alphas[i] > 0):

                    j = self.get_rand_j(i, n)

                    f_xj = float(np.multiply(self.alphas, self.y).T * self.kernel(self.X, self.X[j, :])) + self.b
                    E_j = f_xj - float(self.y[j])

                    prev_alpha_i = self.alphas[i].copy()
                    prev_alpha_j = self.alphas[j].copy()

                    if self.y[i] != self.y[j]:
                        (L, H) = max(0, self.alphas[j] - self.alphas[i]), min(self.regularization, self.regularization + self.alphas[j] - self.alphas[i])
                    else:
                        (L, H) = max(0, self.alphas[j] + self.alphas[i] - self.regularization), min(self.regularization, self.alphas[i] + self.alphas[j])

                    if L == H:
                        continue

                    eta = 2.0 * self.kernel(self.X[i, :], self.X[j, :]) - self.kernel(self.X[i, :], self.X[i, :]) - self.kernel(self.X[j, :], self.X[j, :])

                    if eta >= 0:
                        continue

                    self.alphas[j] = prev_alpha_j - self.y[j] * (E_i - E_j) / eta

                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    if self.alphas[j] < L:
                        self.alphas[j] = L

                    if abs(self.alphas[j] - prev_alpha_j) < 0.00001:
                        continue

                    self.alphas[i] = prev_alpha_i + self.y[i] * self.y[j] * (prev_alpha_j - self.alphas[j])

                    b1 = self.b - E_i - self.y[i] * (self.alphas[i] - prev_alpha_i) * self.kernel(self.X[i, :], self.X[i, :]) - self.y[j] * (self.alphas[j] - prev_alpha_j) * self.kernel(self.X[i, :], self.X[j, :])
                    b2 = self.b - E_j - self.y[i] * (self.alphas[i] - prev_alpha_i) * self.kernel(self.X[i, :], self.X[j, :]) - self.y[j] * (self.alphas[j] - prev_alpha_j) * self.kernel(self.X[j, :], self.X[j, :])

                    if 0 < self.alphas[i] < self.regularization:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.regularization:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                    num_changed_alphas += 1
                    break
                if num_changed_alphas == 0:
                    passes += 1
                else:
                    passes = 0

    def predict(self, X):
        X = SVM.normalize(X)
        self.compute_w()
        return np.sign(self.w.reshape(1, -1) @ X.T + self.b).reshape(-1, 1)

    def get_rand_j(self, i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    def compute_w(self):
        self.w = np.zeros((self.X.shape[1], 1))
        for i in range(len(self.X)):
            self.w += np.multiply(self.alphas[i] * self.y[i], self.X[i, :].T)

    def info(self):
        print("############ SVM's params ################")
        print("Regularization:", self.regularization)
        print("Max iterations:", self.max_iteration)
        print("Learning rate:", self.learning_rate)
        print("Margin of tolerance :", self.tol)
        print("kernel_type:", self.kernel_type)
        print("############ ------------ ################")

    @staticmethod
    def normalize(X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return StandardScaler().fit_transform(X)