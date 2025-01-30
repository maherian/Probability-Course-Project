import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

# Parameters for the problem
k1, k2, k3, k4 = 0.4, 0.6, 0.015, 0.002  # Coefficients in the strength equation
mu_c, sigma_c = 50, 5  # Mean and standard deviation of cement quality
mu_a, sigma_a = 30, 3  # Mean and standard deviation of aggregate quality
sigma_epsilon = 2  # Noise standard deviation
mu_T = 0.5  # Updated mean curing time for exponential distribution
S_min = 30  # Critical strength threshold

# Variance and standard deviation of S_t
sigma_S = np.sqrt((k1 * sigma_c) ** 2 + (k2 * sigma_a) ** 2 + sigma_epsilon ** 2)

# Probability P(S_t < S_min) for a fixed T_c = t
def P_S_t_less_S_min(t):
    mu_S_t = k1 * mu_c + k2 * mu_a + k3 * t - k4 * t**2
    return norm.cdf((S_min - mu_S_t) / sigma_S)

# PDF of T_c with mean mu_T
def pdf_T_c(t):
    return (1 / mu_T) * np.exp(-t / mu_T) if t >= 0 else 0

# Combined integrand for the total probability
def integrand(t):
    return P_S_t_less_S_min(t) * pdf_T_c(t)

# Perform numerical integration over t in [0, ∞)
P_defect, _ = quad(integrand, 0, np.inf)

print(f"P(defect) = {P_defect:.6f}")