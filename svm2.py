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

        self.alphas = None
        self.w = None
        self.b = None
        np.random.seed(1234)

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
        self.X = np.mat(SVM.normalize(X))
        n = len(X)
        self.y = np.mat(y).T
        self.b = 0.0
        self.alphas = np.mat(np.zeros((n, 1)))
        passes = 0
        max_passes = 10

        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(n):

                fXi = float(np.multiply(self.alphas, self.y).T * (self.X * self.X[i, :].T)) + self.b
                Ei = fXi - float(y[i])
                if ((self.y[i] * Ei < -self.tol) and (self.alphas[i] < self.regularization)) or ((self.y[i] * Ei > self.tol) and (self.alphas[i] > 0)):

                    j = self.get_rand_j(i, n)

                    fXj = float(np.multiply(self.alphas, self.y).T * (self.X * self.X[j, :].T)) + self.b
                    Ej = fXj - float(self.y[j])

                    alphaIold = self.alphas[i].copy()
                    alphaJold = self.alphas[j].copy()

                    if self.y[i] != self.y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.regularization, self.regularization + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[j] + self.alphas[i] - self.regularization)
                        H = min(self.regularization, self.alphas[j] + self.alphas[i])

                    if L == H:
                        continue

                    eta = 2.0 * self.X[i, :] * self.X[j, :].T - self.X[i, :] * self.X[i, :].T - self.X[j, :] * self.X[j, :].T

                    if eta >= 0:
                        continue

                    self.alphas[j] -= self.y[j] * (Ei - Ej) / eta

                    self.alphas[j] = self.clipAlphasJ(self.alphas[j], H, L)

                    if abs(self.alphas[j] - alphaJold) < 0.00001:
                        continue

                    self.alphas[i] += self.y[j] * self.y[i] * (alphaJold - self.alphas[j])

                    b1 = self.b - Ei - self.y[i] * (self.alphas[i] - alphaIold) * self.X[i, :] * self.X[i, :].T - self.y[j] * (self.alphas[j] - alphaJold) * self.X[i, :] * self.X[j, :].T
                    b2 = self.b - Ej - self.y[i] * (self.alphas[i] - alphaIold) * self.X[i, :] * self.X[j, :].T - self.y[j] * (self.alphas[j] - alphaJold) * self.X[j, :] * self.X[j, :].T

                    if (0 < self.alphas[i]) and (self.regularization > self.alphas[i]):
                        self.b = b1
                    elif (0 < self.alphas[j]) and (self.regularization > self.alphas[j]):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                    num_changed_alphas += 1
                    # break
                if num_changed_alphas == 0:
                    passes += 1
                else:
                    passes = 0

    def predict(self, X):
        X = SVM.normalize(X)
        self.computeW()
        return np.sign(self.w.reshape(1, -1) @ X.T + self.b).reshape(-1, 1)
        # pred = np.array([])
        # for i in range(len(X)):
        #     f = self.w.reshape(1, -1) @ X[i, :].reshape(-1, 1) + self.b
        #     if f >= 0:
        #         pred = np.append(pred, 1)
        #     else:
        #         pred = np.append(pred, -1)
        # return pred
        # return np.sign(self.w @ X.T + self.b)
        # return self.f(X, self.w, self.b)

    def get_rand_j(self, i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    def clipAlphasJ(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def computeW(self):
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