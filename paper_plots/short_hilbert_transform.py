# ----------------------------------------------------------------------------------------------------------------------
# This module produces plots for evaluating the Hilbert transform for harmonic signals
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 08.00.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import hilbert, lfilter, freqz

from pathlib import Path
import os
from tqdm import tqdm


def use_latex():
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "xelatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


SAVE_PLOTS = True

if SAVE_PLOTS:
    use_latex()


def plot_STHT():
    # find the directory for this file
    root = os.path.join(Path(__file__).resolve().parent, "STHT_plots")

    if not os.path.exists(root):
        os.mkdir(root)

    # parameters of the system
    fs = 48_000
    freq_list = np.asarray([500, 1_000, 2_000, 4_000])

    win_duration = 2 / np.min(freq_list)
    win_len = int(fs * win_duration)
    win_len = 2 * (win_len // 2) + 1

    impulse = np.zeros(win_len)
    impulse[0] = 1
    ker_h = np.fft.fftshift(np.imag(hilbert(impulse)))

    # signal duration
    sig_duartion = 4 / np.min(freq_list)
    sig_len = int(fs * sig_duartion)
    time_vec = np.linspace(0, sig_duartion, sig_len)

    # compute the frequency response
    num_freq_samples = 10_000
    w, h = freqz(ker_h, a=1, worN=num_freq_samples, whole=True, plot=None, fs=fs, include_nyquist=True)
    w[w >= fs / 2] = w[w >= fs / 2] - fs

    # print the kernel
    plt.figure(figsize=(6, 6.5))
    filename = os.path.join(root, "kernel.pgf")

    plt.subplot(211)
    plt.plot(np.arange(0, len(ker_h)) / fs * 1000, ker_h)
    plt.xlabel("time (ms)")
    plt.ylabel("STHT Kernel")
    plt.title(fr"STHT kernel: duration: {int(win_duration * 1000):0.1f} ms, $f_s$: {int(fs / 1000)} KHz")
    plt.grid(True)

    plt.subplot(212)
    plt.plot(w / 1000, 20 * np.log10(np.abs(h)))
    plt.ylim([-5, 5])
    plt.ylabel("STHT freq response [dB]")
    plt.xlabel("freq (KHz)")
    plt.grid(True)

    if SAVE_PLOTS:
        plt.savefig(filename)
    else:
        plt.draw()

    plt.show()

    for freq in tqdm(freq_list):
        filename = os.path.join(root, f"freq={freq}Hz.pgf")
        plt.figure()

        sig_in = np.sin(2 * np.pi * freq * time_vec)
        sig_h = sig_in + 1j * lfilter(ker_h, [1], sig_in)

        sig_hilbert = hilbert(sig_in)

        plt.plot(time_vec, np.real(sig_h), label="STHT-I <= input signal")
        plt.plot(time_vec, np.imag(sig_h), label="STHT-Q")
        plt.plot(time_vec, np.imag(sig_hilbert), label="Hilbert-Q")
        plt.legend()
        plt.grid(True)
        plt.xlabel("time (sec)")
        plt.ylabel("signals")
        plt.title(f"STHT: f: {freq} Hz, {r'$f_s$'}: {int(fs / 1000)} KHz, kernel-dur: {1000 * win_duration:0.1f} ms")

        if SAVE_PLOTS:
            plt.savefig(filename)
        else:
            plt.draw()

        print(f"simulation was done for freq: {freq} and plot was saved in file:{filename}")

    if SAVE_PLOTS:
        pass
    else:
        plt.show()


def main():
    plot_STHT()


if __name__ == '__main__':
    main()
