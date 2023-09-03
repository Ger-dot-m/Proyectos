import numpy as np
from scipy.interpolate import CubicSpline
from ripser import ripser

class sliding_window:
    def __init__(self):
        # For sliding window
        self.tau = None
        self.d = None
        # For persistent homology
        self.SW = None
        self.last_dim = None
    
    def fit_transform(self, f, tau, d, n_data):
        # Step 1: Turn f into a cubic spline
        self.tau = tau
        self.d = d
        if f.shape[0] == 1:
            # If only y values are given, assume x values are 0, 1, 2, ...
            x_values = np.arange(f.shape[1])
            f = np.vstack((x_values, f))
        cs = CubicSpline(f[0], f[1])

        # Step 2: Create the t values where to evaluate SW_f
        t_values = np.linspace(f[0, 0], f[0, -1] - (self.d - 1) * self.tau, n_data)

        # Step 3: Evaluate the sliding window point cloud
        SW = np.zeros((n_data, self.d + 1))
        for i, t in enumerate(t_values):
            t_window = t + np.arange(self.d) * self.tau
            x_values = cs(t_window)
            SW[i] = np.concatenate(([t], x_values))
        self.SW = SW
        return SW
    
    def max_pers(self, SW, dim = 1, n_landmarks = 200, prime_coeff = 7):
        # Inputs:
        self.SW = SW
        self.last_dim = dim
        #Step 1: Compute the Vietoris-Rips persistence diagram
        diagrams = ripser(SW, n_perm = n_landmarks, coeff = prime_coeff, maxdim=dim)['dgms']
        #Step 2: Extract the 1-dimensional persistence pairs
        pairs = diagrams[dim]
    
        #Step 3: Compute mp_dim
        mp_1 = 0
        for (a, b) in pairs:
            persistence = b - a
            mp_1 = max(mp_1, persistence)
    
        return mp_1
    
    def score(self, SW, dim = 1, n_landmarks = 200, prime_coeff = 7):
        a = self.max_pers(SW, dim = dim, n_landmarks=n_landmarks, prime_coeff=prime_coeff)
        b = np.sqrt(3)
        return a/b
        