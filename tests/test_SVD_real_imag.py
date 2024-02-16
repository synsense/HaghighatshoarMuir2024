# ----------------------------------------------------------------------------------------------------------------------
# This module checks the relation between real and complex SVD in the case of beamformer computation.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 22.09.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hilbert


def test_SVD():
    dim = 30
    duration = 1
    index = 0

    # build a filter
    order = 2
    fs = 48_000
    cutoff = [1000, 10_000]
    T = int(duration * fs)

    b, a = butter(order, cutoff, btype="pass", output="ba", analog=False, fs=fs)

    # complex version
    sig = lfilter(b, a, np.random.rand(dim, T), axis=1)
    sig_h = hilbert(sig)

    sig_in = sig + 1j * sig_h

    # compute SVD
    cov = 1 / T * (sig_in @ sig_in.conj().T)
    U, D, _ = np.linalg.svd(cov)
    u = np.hstack([np.real(U[:, index]), np.imag(U[:, index])])

    # real version
    sig_in_r = np.vstack([np.real(sig_in), np.imag(sig_in)])
    cov_r = 1 / T * (sig_in_r @ sig_in_r.T)
    U_r, D_r, _ = np.linalg.svd(cov_r)
    u_r = U_r[:, index]

    # plt.plot(u, label="complex")
    # plt.plot(u_r, label="real")
    plt.plot(D, ".", label="complex")
    plt.plot(np.diff(np.cumsum(D_r)[2::2], prepend=0), ".", label="real")
    plt.legend()
    plt.title("singular values for real and imaginary case")
    plt.xlabel("dim")
    plt.ylabel("eigen vector")
    plt.grid(True)
    plt.show()


def main():
    test_SVD()


if __name__ == "__main__":
    main()
