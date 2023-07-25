# ----------------------------------------------------------------------------------------------------------------------
# This module investigates how the phase of a given signal varies after Hilbert transform.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.07.2023
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.signal import hilbert, lfilter, butter, medfilt
import matplotlib.pyplot as plt
from numpy.linalg import norm

def test_hilbert_phase():
    # a random signal
    noise = np.random.randn(100)
    fs = 50_000

    # filter it
    order = 2
    cutoff = [1_000, 12_000]

    b, a = butter(order, cutoff, btype='bandpass', output='ba', analog=False, fs=fs)
    sig_in = lfilter(b, a, noise)


    sig_in_h = hilbert(sig_in)

    phase = np.unwrap(np.angle(sig_in_h))
    

    plt.subplot(311)
    plt.plot(phase)
    plt.grid(True)
    plt.ylabel("phase of hilbert transform")
    plt.title(f"phase per sample: {phase[-1]/len(phase) * 180 / np.pi:0.0f} degree")

    plt.subplot(312)
    plt.plot(np.real(sig_in_h), label="original")
    plt.plot(np.imag(sig_in_h), label="hilbert")
    plt.plot(np.abs(sig_in_h), 'k', label="envelope")
    plt.plot(-np.abs(sig_in_h), 'k')
    plt.xlabel("time")
    plt.ylabel("envelope")
    plt.legend()
    plt.grid(True)

    plt.subplot(313)
    sig_mod = np.linspace(1, 2, len(phase)) * np.exp(1j*phase)
    plt.plot(np.real(sig_mod), np.imag(sig_mod), '.')

    plt.show()


def test_hilbert_phase_spike():
    # a spike signal
    sig_len = 100
    prob = 0.9

    support = (np.random.rand(sig_len) < prob) * 1.0
    spike = np.sign(np.random.randn(sig_len)) * support

    spike_h = hilbert(spike)
    phase = np.unwrap(np.angle(spike_h))

    plt.subplot(211)
    plt.plot(np.real(spike_h))
    plt.plot(np.imag(spike_h))
    plt.grid(True)

    plt.subplot(212)
    plt.plot(phase)
    plt.ylabel("phase")

    plt.show()


def test_inst_frequency_chirp():
    """
    this module checks if Hilbert transform is able to extract the instantenous frequency of a signal.
    """

    # chirp signal
    fs = 50_000
    fmin = 1_000
    fmax = 10_000
    period = 1

    time_vec = np.arange(0, period, step=1/fs)
    freq_inst = fmin + (fmax - fmin) * (time_vec / period)

    phase = 2*np.pi * np.cumsum(freq_inst) / fs
    sig_in = np.sin(phase)

    sig_in_h = hilbert(sig_in)

    phase_est = np.unwrap(np.angle(sig_in_h))
    freq_inst_est = 1/(2*np.pi) * np.diff(phase_est) * fs

    rel_err = np.sqrt(np.median((freq_inst[:-1] - freq_inst_est)**2))/ np.sqrt(np.median(freq_inst_est**2) * np.median(freq_inst**2))

    # now use limited span Hilbert kernel for estimation
    kernel_duration = 10 / fmin
    kernel_len = int(fs * kernel_duration)

    impulse = np.zeros(kernel_len)
    impulse[0] = 1
    kernel = np.imag(hilbert(impulse))

    sig_in_h_ker = sig_in + 1j * lfilter(kernel, [1], sig_in)
    phase_ker = np.unwrap(np.angle(sig_in_h_ker))
    freq_inst_ker = 1 / (2 * np.pi) * np.diff(phase_ker) * fs


    plt.subplot(211)
    plt.plot(freq_inst, label="original chirp freq")
    plt.plot(freq_inst_est, label="estimated chirp freq")
    plt.plot(freq_inst_ker, label="kernel chirp freq")
    plt.title(f"inst freq => relative error between original and estimate: {rel_err:e}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("time")
    plt.ylabel("instanteneous frequency")

    plt.subplot(212)
    plt.plot(phase, label="original chirp freq")
    plt.plot(phase_est, label="estimated chirp freq")
    plt.plot(phase_ker, label="kernel chirp freq")
    plt.title(f"phase signal")
    plt.legend()
    plt.grid(True)
    plt.xlabel("time")
    plt.ylabel("instanteneous frequency")

    plt.show()


def test_inst_freq_mixture():
    # signal characteristic
    fs = 50_000
    num_freq = 3
    freq_vec = 10_000 * np.random.rand(num_freq)
    phase_vec = 2*np.pi * np.random.rand(num_freq)

    num_periods = 100
    duartion = num_periods / np.min(freq_vec)

    time_vec = np.arange(0, duartion, step=1/fs)

    sig_in = np.sum(np.sin(2*np.pi*freq_vec.reshape(-1, 1) * time_vec.reshape(1,-1) + phase_vec.reshape(-1,1)), axis=0)

    sig_in_h = hilbert(sig_in)
    phase = np.unwrap(np.angle(sig_in_h))

    freq_inst = 1/(2*np.pi) * np.diff(phase) * fs

    # apply smoothing for better result
    kernel_size = 1
    freq_inst = medfilt(freq_inst, kernel_size=kernel_size)

    fmin = np.min(freq_vec)
    fmax = np.max(freq_vec)

    # now use limited span Hilbert kernel for estimation
    kernel_duration = 10/np.max(freq_vec)
    kernel_len = int(fs*kernel_duration)

    impulse = np.zeros(kernel_len)
    impulse[0] = 1
    kernel = np.imag(hilbert(impulse))

    sig_in_h_ker = sig_in + 1j*lfilter(kernel, [1], sig_in)
    phase_ker = np.unwrap(np.angle(sig_in_h_ker))
    freq_inst_ker = 1 / (2 * np.pi) * np.diff(phase_ker) * fs

    freq_inst_ker = medfilt(freq_inst_ker, kernel_size=kernel_size)

    rel_err = norm(freq_inst - freq_inst_ker)/ np.sqrt(norm(freq_inst) * norm(freq_inst_ker))

    plt.plot(time_vec[:-1], freq_inst, label="inst freq")
    plt.plot(time_vec[:-1], freq_inst_ker, label="inst freq kernel")
    plt.axhline(y=fmin, color='b', linestyle='-', label='freq min')
    plt.axhline(y=fmax, color='r', linestyle='-', label='freq max')
    plt.title(f"inst freq estimation: median filter kernel size: {kernel_size}, rel-err: {rel_err}")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("frequency")
    plt.grid(True)
    plt.show()




def main():
    # test_hilbert_phase()
    # test_hilbert_phase_spike()
    test_inst_frequency_chirp()
    # test_inst_freq_mixture()


if __name__ == '__main__':
    main()