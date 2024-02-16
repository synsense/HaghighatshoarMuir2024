# ----------------------------------------------------------------------------------------------------------------------
# This module checks how the length of the window in Hilbert transform affects the frequency response.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.07.2023
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.signal import lfilter, hilbert
from tqdm import tqdm
import matplotlib.pyplot as plt


def test_freq_response():
    # signal specs
    fs = 50_000
    ker_duration = 10e-3
    sig_duration = 1

    ker_len = int(fs * ker_duration)

    # build the Hilbert kernel
    impulse = np.zeros(ker_len)
    impulse[0] = 1

    ker_h = np.fft.fftshift(np.imag(hilbert(impulse)))

    # range of frequencies to be covered
    fmin_ker = 2 / ker_duration
    fmin = min([1 * fmin_ker, fs / 2])
    fmax = min([150 * fmin_ker, fs / 2])

    num_freq = 1000
    freq_vec = 10 ** np.linspace(np.log10(fmin), np.log10(fmax), num_freq)

    freq_res = []
    time_vec = np.arange(0, sig_duration, step=1 / fs)

    for freq in tqdm(freq_vec):
        sig_in = np.sin(2 * np.pi * freq * time_vec)
        sig_in_h = lfilter(ker_h, [1], sig_in)

        # take the stable part
        sig_in_h_stable = sig_in_h[ker_len:]
        sig_in_h_stable = sig_in_h[len(sig_in_h) // 4 :]

        freq_response = np.sqrt(np.mean(sig_in_h_stable**2) / np.mean(sig_in**2))

        freq_res.append(freq_response)

    freq_res = np.asarray(freq_res)
    freq_res_dB = 20 * np.log10(freq_res)

    # plot the results
    plt.figure(figsize=(16, 10))
    plt.plot(freq_vec, freq_res_dB)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("frequency response")
    plt.grid(True)
    plt.title(
        f"ker-duration={ker_duration}, fs={fs}, fmin={fmin/1000:0.2f} KHz, fmax={fmax/1000:0.1f} KHz"
    )
    plt.show()


def main():
    test_freq_response()


if __name__ == "__main__":
    main()
