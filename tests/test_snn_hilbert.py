# ----------------------------------------------------------------------------------------------------------------------
# This module investigates the effect of filtering on phase of spike signal.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 04.09.2023
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.signal import hilbert, lfilter, butter
import matplotlib.pyplot as plt


def test_spike_phase():
    # specification of STHT
    ker_duration = 10e-3
    fs = 48_000
    ker_len = int(fs * ker_duration)

    impulse = np.zeros(ker_len)
    impulse[0] = 1
    ker_h = np.fft.fftshift(np.imag(hilbert(impulse)))

    # build a filter
    spike_rate = 1234
    cut_off = [0.005 / ker_duration, 3_000]
    b, a = butter(2, cut_off, analog=False, btype="pass", output='ba', fs=fs)

    # cancel filtering stage
    # b = [1]
    # a = [1]

    # build the spike signal
    sig_duration = 1
    time_vec = np.arange(0, sig_duration, step=1/fs)

    sig_len = int(fs * sig_duration)
    num_spike = int(sig_duration * spike_rate)

    spike = np.zeros(sig_len)
    spike[np.argsort(np.random.randn(sig_len))[:num_spike]] = 1


    # sinusoid model for spike
    freq_min = 1100
    freq_max = 2000
    period = 0.5

    freq_vec = freq_min + (freq_max - freq_min) * (time_vec % period)/period
    phase = 2 * np.pi * np.cumsum(freq_vec)/fs
    sig_in = np.sin(phase)
    threshold = np.sum(np.abs(sig_in))/num_spike

    spike = np.diff(np.floor(np.cumsum(np.abs(sig_in))/threshold))

    spike = np.zeros_like(sig_in)
    spike[:-1][np.sign(sig_in[:-1] * sig_in[1:] * np.diff(sig_in)) > 0] = 1


    # filtered spike signal
    spike_filt = lfilter(b, a, spike)

    # apply STHT and build the phase
    spike_filt_h = spike_filt + 1j * lfilter(ker_h, [1], spike_filt)
    phase = np.unwrap(np.angle(spike_filt_h))



    plt.plot(time_vec[:len(phase)], phase)
    plt.xlabel("time (sec)")
    plt.ylabel("unwrapped phase")
    plt.grid(True)
    plt.title("phase plot")
    plt.show()


def main():
    test_spike_phase()


if __name__ == '__main__':
    main()
