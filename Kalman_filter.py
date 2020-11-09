import numpy as np
import math, cmath

class Kalman_filter:
    def __init__(self, x_0, P_0, Q, R):
        self.x_k = x_0
        self.P_k = P_0
        self.Q = Q
        self.R = R

    def __estimate(self, A, H):
        self.x_k_ = A * self.x_k
        self.P_k_ = A * self.P_k * A.T + self.Q
        
        self.K_k = self.P_k_ * H.T * np.linalg.inv(H * self.P_k_ * H.T + R)

    def __calculate(self, z, H):
        self.x_k = self.x_k_ + self.K_k * (z - H* self.x_k_)
        self.P_k = self.P_k_ - self.K_k * H * self.P_k_

        return self.x_k

    def process(self, z, A, H):
        self.__estimate(A, H)
        return self.__calculate(z, H)
