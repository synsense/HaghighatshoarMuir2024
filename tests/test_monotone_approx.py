# ----------------------------------------------------------------------------------------------------------------------
# This module finds the best monotone approximation of a function.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 26.09.2023
# ----------------------------------------------------------------------------------------------------------------------


import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


def test_monotone():
    dim = 100
    snr_db = 20
    snr = 10 ** (snr_db / 10)

    sig_in = np.sort(np.random.rand(100))
    sig_in_noise = sig_in + np.sqrt(np.mean(sig_in**2))/np.sqrt(snr) * np.random.randn(dim)

    # Construct the problem.
    x = cp.Variable(dim)
    # objective = cp.Minimize(cp.sum(cp.abs(sig_in_noise - x)))
    objective = cp.Minimize(cp.sum_squares(sig_in_noise - x))
    constraints = [x[1:] >= x[:-1]]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    sig_est = x.value
    print("estimated signal: ", sig_est)

    plt.figure()
    plt.plot(sig_in_noise, label="noisy")
    plt.plot(sig_in, label="original")
    plt.plot(sig_est, label="estimate")
    plt.legend()

    plt.show()



def main():
    test_monotone()


if __name__ == '__main__':
    main()
