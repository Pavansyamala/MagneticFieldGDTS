import numpy as np 

import matplotlib.pyplot as plt


class TransmitterFrameField:

    def __init__(self, pr , pt):
        self.m = [1,0,0]
        
        R = np.subtract(np.transpose(pr), pt).T
        norm_R = np.sqrt(np.einsum("i...,i...", R, R))

        # Calculate A (replace x, y, z with appropriate values)
        x, y, z = pr
        A = np.array([2*x*2 - y**2 - z*2, 3*x*y, 3*x*z])

        # Compute magnetic field B
        norm_m = np.sqrt(np.einsum("i...,i...", self.m, self.m))
        self.B = np.array((norm_m / ((4 * np.pi) * norm_R**5)) * A)