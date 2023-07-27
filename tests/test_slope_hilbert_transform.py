# ----------------------------------------------------------------------------------------------------------------------
# This module checks how the phase slope varies in Hilbert transform.
#
#
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 27.07.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, lfilter


def test_slope_hilbert():
    """
    this module checks how the slope of the phase of Hilbert transform affects the localization precision.
    """
    # signal specs
    fs = 50_000
    kernel_duration = 10e-3
    kernel_len = int(kernel_duration * fs)

    sig_duration = 100e-3
    sig_len = int(fs * sig_duration)

    impulse = np.zeros(kernel_len)
    impulse[0] = 1
    kernel = np.imag(hilbert(impulse))

    # binary signal
    ones_ratio = 0.1
    num_ones = int(ones_ratio * sig_len)
    permutation = np.argsort(np.random.randn(sig_len))

    b = np.zeros(sig_len)
    b[permutation[:num_ones]] = 1

    bh = hilbert(b)
    phase = np.unwrap(np.angle(bh))
    slope = np.mean(np.diff(phase))

    bker = b + 1j * lfilter(kernel, [1], b)
    phase_ker = np.unwrap(np.angle(bker))
    slope_ker = np.mean(np.diff(phase_ker))

    print("hilbert slope for a binary signal: ", slope)
    print("hilbert kernel slope for a binary signal: ", slope_ker)

    # antipodal signal
    x = 2 * b - 1

    xh = hilbert(x)
    phase = np.unwrap(np.angle(xh))
    slope = np.mean(np.diff(phase))

    xker = x + 1j * lfilter(kernel, [1], x)
    phase_ker = np.unwrap(np.angle(xker))
    slope_ker = np.mean(np.diff(phase_ker))

    print("hilbert slope for an anti-podal signal: ", slope)
    print("hilbert kernel slope for an anti-podal signal: ", slope_ker)

    plt.plot(phase, label="hilbert")
    plt.plot(phase_ker, label="hilbert kernel")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    test_slope_hilbert()


if __name__ == '__main__':
    main()
