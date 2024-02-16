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
from micloc.spike_encoder import ZeroCrossingSpikeEncoder

from pathlib import Path
import os
from tqdm import tqdm


def use_latex():
    matplotlib.use("pdf")
    matplotlib.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.5,
            #     "pgf.texsystem": "xelatex",
            #     "font.family": "Helvetica",
            #     "text.usetex": True,
            #     "pgf.rcfonts": False,
        }
    )

    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=SMALL_SIZE)  # fontsize of the figure title


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
    sig_duration = 4 / np.min(freq_list)
    sig_len = int(fs * sig_duration)
    time_vec = np.linspace(0, sig_duration, sig_len)

    # compute the frequency response
    num_freq_samples = 10_000
    w, h = freqz(
        ker_h,
        a=1,
        worN=num_freq_samples,
        whole=True,
        plot=None,
        fs=fs,
        include_nyquist=True,
    )
    w[w >= fs / 2] = w[w >= fs / 2] - fs

    # print the kernel
    mm = 1 / 25.4

    plt.figure(figsize=[70 * mm, 90 * mm])
    filename = os.path.join(root, "kernel.pdf")

    plt.subplot(211)
    plt.plot(np.arange(0, len(ker_h)) / fs * 1000, ker_h)
    plt.plot([0, len(ker_h) / fs * 1000], [0, 0], "k:")
    plt.xlabel("Time (ms)")
    plt.ylabel("STHT Kernel")
    plt.title(
        rf"STHT kernel: Duration {int(win_duration * 1000):0.1f} ms, $F_s$: {int(fs / 1000)} KHz"
    )
    plt.grid(False)

    plt.subplot(212)
    plt.plot(w / 1000, 20 * np.log10(np.abs(h)))
    plt.ylim([-5, 5])
    plt.ylabel("STHT response (dB)")
    plt.xlabel("F (KHz)")
    plt.grid(False)

    if SAVE_PLOTS:
        plt.savefig(filename, bbox_inches="tight", transparent=True)
    else:
        plt.draw()

    plt.show()

    for freq in tqdm(freq_list):
        filename = os.path.join(root, f"freq={freq}Hz.pdf")

        mm = 1 / 25.4

        plt.figure(figsize=[70 * mm, 45 * mm])

        sig_in = np.sin(2 * np.pi * freq * time_vec) + np.random.normal(
            scale=0.025, size=len(time_vec)
        )
        sig_h = sig_in + 1j * lfilter(ker_h, [1], sig_in)

        sig_hilbert = hilbert(sig_in)

        plt.plot(time_vec * 1e3, np.real(sig_h), label="STHT-I <= input signal")
        plt.plot(time_vec * 1e3, np.imag(sig_h), color="C1", label="STHT-Q")
        plt.plot(
            time_vec * 1e3, np.imag(sig_hilbert), "--", color="black", label="Hilbert-Q"
        )
        plt.plot(time_vec[[0, -1]] * 1e3, [0, 0], "k:")
        # plt.legend()
        plt.grid(False)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        # plt.title(
        #     f"STHT: f: {freq} Hz, {r'$f_s$'}: {int(fs/1000)} KHz, kernel-dur: {1000*win_duration:0.1f} ms"
        # )

        if SAVE_PLOTS:
            plt.savefig(filename, bbox_inches="tight", transparent=True)
        else:
            plt.draw()

        # Make a spike encodder
        freq_range = [freq / 2, 2 * freq]
        zc_dist = int(fs / freq_range[-1])
        robust_width = zc_dist // 2

        print(robust_width)
        rzcc = ZeroCrossingSpikeEncoder(fs, robust_width)

        plt.figure(figsize=[70 * mm, 5 * mm])
        sig_cat = np.stack([np.real(sig_h), np.imag(sig_h)])
        spikes = rzcc.evolve(sig_cat.T)
        # spikes[spikes < 0] = 0

        # plt.plot(spikes)
        plt.stem(time_vec * 1e3, -spikes[:, 0], "C0-", markerfmt="", basefmt="none")
        plt.stem(time_vec * 1e3, -spikes[:, 1], "C1-", markerfmt="", basefmt="none")
        plt.grid(False)
        plt.xlabel("Time (ms)")
        plt.yticks([])
        plt.axis("off")
        # plt.ylabel("Amplitude")

        filename = os.path.join(root, f"spikes_freq={freq}Hz.pdf")
        plt.savefig(f"{filename}", bbox_inches="tight", transparent=True)

        print(
            f"simulation was done for freq: {freq} and plot was saved in file:{filename}"
        )

    if SAVE_PLOTS:
        pass
    else:
        plt.show()


def main():
    plot_STHT()


if __name__ == "__main__":
    main()
