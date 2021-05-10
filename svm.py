"""
Ref: https://aihubprojects.com/svm-from-scratch-python/
     http://cs229.stanford.edu/materials/smo.pdf
     https://ai6034.mit.edu/wiki/images/SVM_and_Boosting.pdf
     https://jonchar.net/notebooks/SVM/
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rn

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

        self.alpha = None
        self.w = None
        self.b = None

    def kernel_linear(self, x1, x2):
        return x1 @ x2.T + self.b

    def kernel_poly(self, x1, x2, degree=2):
        if degree == 1:
            self.kernel_linear(x1, x2)
        elif degree > 1:
            return (x1 @ x2.T + self.b) ** degree
        else:
            print('Incorrect degree')

    def kernel_rbf(self, x1, x2, sigma=1):
        if np.ndim(x1) == 1 and np.ndim(x2) == 1:
            result = np.exp(- (np.linalg.norm(x1 - x2, 2)) ** 2 / (2 * sigma ** 2))
        elif (np.ndim(x1) > 1 and np.ndim(x2) == 1) or (np.ndim(x1) == 1 and np.ndim(x2) > 1):
            result = np.exp(- (np.linalg.norm(x1 - x2, 2, axis=1) ** 2) / (2 * sigma ** 2))
        elif np.ndim(x1) > 1 and np.ndim(x2) > 1:
            result = np.exp(-(np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
        return result

    def train(self, X, y):
        X = SVM.normalize(X)
        n = len(X)
        self.alpha = np.zeros(n)
        self.b = 0.0
        iteration = 0

        while True:
            iteration += 1
            prev_alpha = np.copy(self.alpha)

            for j in range(0, n):
                # get random i is not equal to j
                i = self.get_rand_j(j, n - 1)

                x_i, y_i = X[i, :], y[i]
                x_j, y_j = X[j, :], y[j]

                eta = self.kernel(x_i, x_i) + self.kernel(x_j, x_j) - 2 * self.kernel(x_i, x_j)
                if eta == 0:
                    continue

                prev_alpha_j, prev_alpha_i = self.alpha[j], self.alpha[i]

                if y_i != y_j:
                    (L, H) = max(0, prev_alpha_j - prev_alpha_i), min(self.regularization, self.regularization + prev_alpha_j - prev_alpha_i)
                else:
                    (L, H) = max(0, prev_alpha_i + prev_alpha_j - self.regularization), min(self.regularization, prev_alpha_i + prev_alpha_j)
                if L == H:
                    continue

                self.w = (self.alpha * y) @ X
                self.b = np.mean(y - self.w @ X.T)

                E_i = self.f(x_i, self.w, self.b) - y_i
                E_j = self.f(x_j, self.w, self.b) - y_j

                # set new alpha values
                self.alpha[j] = prev_alpha_j + float(y_j * (E_i - E_j)) / eta
                if self.alpha[j] > H:
                    self.alpha[j] = H
                elif self.alpha[j] < L:
                    self.alpha[j] = L

                self.alpha[i] = prev_alpha_i + y_i * y_j * (prev_alpha_j - self.alpha[j])

            diff = np.linalg.norm(self.alpha - prev_alpha)
            if diff < self.tol:
                break

            if iteration >= self.max_iteration:
                print("Max iterations reached")
                return

        self.b = np.mean(y - self.w @ X.T)
        if self.kernel_type == 'linear':
            self.w = self.kernel_linear(self.alpha * y, X.T)
        elif self.kernel_type == 'poly':
            self.w = self.kernel_poly(self.alpha * y, X.T, degree=2)
        elif self.kernel_type == 'rbf':
            self.w = self.kernel_rbf(self.alpha * y, X.T, sigma=1)

    def predict(self, X):
        X = SVM.normalize(X)
        return self.f(X, self.w, self.b)

    def f(self, X, w, b):
        f_x = w @ X.T + b
        return np.sign(f_x)

    def get_rand_j(self, i, n):
        j = 0
        count = 0
        while j == i and count < 1000:
            j = rn.randint(0, n)
            count += 1
        return j

    def info(self):
        print("############ SVM's params ################")
        print("Regularization:", self.regularization)
        print("Max iterations:", self.max_iteration)
        print("Learning rate:", self.learning_rate)
        print("Margin of tolerance :", self.tol)
        print("kernel_type:", self.kernel_type)

    @staticmethod
    def normalize(X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return StandardScaler().fit_transform(X)