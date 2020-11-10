import numpy as np
import math, cmath

class Kalman_filter:
    def __init__(self, x_0, P_0, Q, R):
        self.x_k = np.matrix(x_0)
        self.P_k = np.matrix(P_0)
        self.Q = np.matrix(Q)
        self.R = np.matrix(R)

    def __estimate(self, A, H):
        self.x_k_ = A * self.x_k
        self.P_k_ = A * self.P_k * A.H + self.Q
        
        self.K_k = self.P_k_ * H.H * np.linalg.inv(H * self.P_k_ * H.H + self.R)

    def __calculate(self, z, H):
        self.x_k = self.x_k_ + self.K_k * (z - H* self.x_k_)
        self.P_k = self.P_k_ - self.K_k * H * self.P_k_

        return self.x_k.T.tolist()[0]

    def process(self, z, A, H):
        z = np.matrix(z)
        A = np.matrix(A)
        H = np.matrix(H)
        self.__estimate(A, H)
        return self.__calculate(z, H)
