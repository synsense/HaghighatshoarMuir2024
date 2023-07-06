# ----------------------------------------------------------------------------------------------------------------------
# This module tests the envelope estimator designed for audio applications.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.07.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from micloc.utils import Envelope


def test_envelope():
    import matplotlib.pyplot as plt
    # a simple sinusoid signal
    freq = 1
    num_period = 20
    duration = num_period / freq
    fs = 100 * freq

    time_in = np.arange(0, duration, step=1 / fs)
    sig_in = np.asarray([np.sin(2 * np.pi * freq * time_in), np.cos(2 * np.pi * freq * time_in)]).T

    num_chan = sig_in.shape[1]

    # envelope extractor module
    rise_time = 3 / freq
    fall_time = 30 / freq

    env = Envelope(rise_time=rise_time, fall_time=fall_time, fs=fs)

    sig_out = env.evolve(sig_in=sig_in)

    plt.figure(figsize=(16, 10))
    for i, sig_in_chan, sig_out_chan in zip(range(num_chan), sig_in.T, sig_out.T):
        plt.subplot(num_chan, 1, i + 1)
        plt.plot(time_in, np.abs(sig_in_chan))
        plt.plot(time_in, np.abs(sig_out_chan))
        plt.legend(["signal", "envelope"])
        plt.grid(True)

        if i == 0:
            plt.title("envelope estimation")

    plt.show()


if __name__ == '__main__':
    test_envelope()
