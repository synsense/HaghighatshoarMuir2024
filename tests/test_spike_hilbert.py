# ----------------------------------------------------------------------------------------------------------------------
# This module checks how localization can be performed by applying Hilbert transform directly to the spikes.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.07.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, hilbert, butter


def test_spike_hilbert():
    # signal characteristics
    fs = 50_000
    freq = 1100
    num_periods = 10
    duration = 310e-3

    time_vec = np.arange(0, duration, step=1 / fs)
    sig_in = np.sin(2 * np.pi * freq * time_vec)

    # use a noisy signal
    order = 2
    cutoff = [0.9*freq, freq]
    btype ='pass'
    
    noise = np.random.randn(len(time_vec))
    b, a = butter(order, cutoff, analog=False, btype=btype, output='ba', fs=fs)
    sig_in = lfilter(b, a, noise)

    target_spike_rate = 500
    threshold = np.mean(np.abs(sig_in)) * fs / target_spike_rate

    spikes = np.diff(np.floor(np.cumsum(np.abs(sig_in)) / threshold))

    kernel_duration = 10e-3
    kernel_length = int(fs * kernel_duration)

    impulse = np.zeros(kernel_length)
    impulse[0] = 1
    kernel_h = np.imag(hilbert(impulse))

    sig_in_h = lfilter(kernel_h, [1], spikes)

    # Hilbert transform
    sig_h = sig_in[:-1] + 1j * sig_in_h

    # apply the Kernel in the filter
    tau = 1.5/(2*np.pi*freq)
    neuron_kernel = time_vec * np.exp(-time_vec/tau)
    neuron_kernel /= np.mean(neuron_kernel)

    sig_h_filtered = lfilter(neuron_kernel, [1], sig_h)

    # compute the phase
    phase = np.unwrap(np.angle(sig_h_filtered))

    plt.figure()
    plt.subplot(311)
    plt.plot(time_vec[:-1], sig_in_h)
    plt.grid(True)
    plt.ylabel("Spike trans")
    plt.title(f"Hilbert transform of spikes: kernel-duration: {kernel_duration:0.4f}, kernel-length:{kernel_length}")

    plt.subplot(312)
    plt.plot(time_vec[:-1], np.real(sig_h_filtered))
    plt.plot(time_vec[:-1], np.imag(sig_h_filtered))
    plt.grid(True)
    plt.ylabel("Smoothed trans")

    plt.subplot(313)
    plt.plot(time_vec[:-1], phase)
    plt.ylabel("phase")

    plt.show()


def main():
    test_spike_hilbert()


if __name__ == '__main__':
    main()
