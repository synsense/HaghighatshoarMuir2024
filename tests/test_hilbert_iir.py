# ----------------------------------------------------------------------------------------------------------------------
# This module tries to see if Hilbert transform can be implemented as an IIR filter.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 17.07.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.signal import hilbert, lfilter, freqz
from scipy.linalg import hankel


def test_hilbert_irr():
    # specification of the system
    fs = 50_000
    kernel_duration = 10e-3

    impulse_len = int(kernel_duration * fs)
    impulse = np.zeros(impulse_len)
    impulse[0] = 1

    kerh = np.imag(hilbert(impulse))
    kerh = kerh[:len(kerh) // 2]

    # decide on MA and AR part degrees
    deg_MA = 6
    deg_AR = 6

    L = len(kerh)
    input = impulse[:L]
    output = kerh[:L]

    hankel_MA = hankel(input[::-1])[::-1, :deg_MA]
    hankel_AR = hankel(output[::-1])[::-1, 1:deg_AR + 1]
    hankel_AR_col = hankel(output[::-1])[::-1, 0]

    A = np.zeros((len(hankel_MA), deg_MA + deg_AR))
    A[:, :deg_MA] = hankel_MA
    A[:, deg_MA:deg_AR + deg_MA] = hankel_AR
    y = hankel_AR_col

    solution = np.linalg.lstsq(A, y, rcond=None)[0]

    # IIR filter parameters
    filt_b = solution[:deg_MA]
    filt_a = np.asarray([1, *(-solution[deg_MA:])])

    print(f"estimated filter parameters:\nb={filt_b}\na={filt_a}\n")

    # compute the impulse response again using the given estimated filter
    kerh_est = lfilter(filt_b, filt_a, input)
    rel_err = norm(kerh - kerh_est)/np.min([norm(kerh), norm(kerh_est)])

    # plot the kernel and its estimate
    plt.subplot(211)
    plt.plot(kerh, label="original kernel")
    plt.plot(kerh_est, label="estimated kernel")
    plt.legend()
    plt.title(fr"ker-len:{L}, deg-MA:{deg_MA}, deg-AR:{deg_AR}, rel-error $\rho$ = {rel_err:0.5f}")
    plt.ylabel("Hilbert transform kernel")
    plt.grid(True)

    # check the relative delay of the filter
    freq, h = freqz(filt_b, filt_a, worN=2**14, fs=fs)
    phase = np.angle(h)
    EPS = 0.0000001
    phase_delay = phase/(2*np.pi*freq + EPS) * 1000

    plt.subplot(212)
    plt.plot(freq, phase_delay)
    plt.ylabel("phase delay [ms]")
    plt.grid(True)
    plt.show()


def main():
    test_hilbert_irr()


if __name__ == '__main__':
    main()
