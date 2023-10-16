# ----------------------------------------------------------------------------------------------------------------------
# This module plots the array resolution for the Hilbert-transform based localization algorithm.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 16.10.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np

from micloc.array_geometry import CenterCircularArray
from micloc.beamformer import Beamformer
from micloc.utils import Envelope
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
from scipy.signal import lfilter, butter
from scipy.linalg import eigh
import os
from pathlib import Path
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
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'xtick.major.size': 2,
            'ytick.major.size': 2,
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



SAVE_PLOTS = False

if SAVE_PLOTS:
    use_latex()


def plot_beampattern(doa_list, corr, title, filename):
    mm = 1 / 25.4

    plt.figure(figsize=[35 * mm, 35 * mm])
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0], polar=True)
    # ax2 = plt.subplot(gs[1], polar=False)

    ax1.plot(doa_list, np.abs(corr[len(corr) // 2]), label="beam pattern")
    ax1.plot(doa_list, np.abs(corr[3 * len(corr) // 4]), label="beam pattern")
    ax1.set_title(title)
    ax1.grid(True)
    ax1.set_xticks(np.arange(0 / 180 * np.pi, 360 / 180 * np.pi, 60 / 180 * np.pi))
    ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels([])

    # selected_indices = np.arange(0, len(corr), len(corr) // 4)
    # selected_doa = doa_list[selected_indices]
    # ax2.plot(doa_list / np.pi * 180, corr[selected_indices, :].T)
    # ax2.legend([f"DoA: {int(rad / np.pi * 180)}" for rad in selected_doa])
    # ax2.set_xlabel("DoA")
    # ax2.set_ylabel("array resolution")
    # ax2.grid(True)

    if SAVE_PLOTS:
        plt.savefig(filename, bbox_inches="tight", transparent=True)
    else:
        plt.draw()


def array_resolution_sin():
    """
    this function computes the array resolution curve at various angle via direct numerical simulation.
    the template signals used for localization are sinusoids.
    """
    # find the directory for this file
    root = os.path.join(Path(__file__).resolve().parent, "array_resolution_sin")

    if not os.path.exists(root):
        os.mkdir(root)

    # build a geometry
    radius = 4.5e-2
    num_mic = 7
    fs = 48_000

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 10e-3


    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 0.4
    freq_design_vec = [1000, 2000, 3600, 4000, 8000]

    for freq_design in tqdm(freq_design_vec):
        print(f"\n\nplotting beam pattern for freq: {freq_design}...")

        filename = os.path.join(root, f"array_resolution_sin_freq={freq_design}.pdf")

        freq_range = [0.8*freq_design, 1.2 * freq_design]
        beamf = Beamformer(geometry=geometry, kernel_duration=kernel_duration, freq_range=freq_range, fs=fs)

        time_temp = np.arange(0, duration, step=1 / fs)
        sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

        # 2. use an angular grid
        num_grid = 32 * num_mic + 1
        doa_list = np.linspace(-np.pi, np.pi, num_grid)

        interference_removal = False
        bf_mat, cov_mat_list = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list,
                                                          interference_removal=interference_removal)

        # a specific DoA
        angle_index = num_grid // 2
        bf_vec = bf_mat[:, angle_index]

        # two different beam patterns considering the covariance matrices of the signals received
        beam_pattern_best = np.asarray([np.abs(bf_vec.conj().T @ cov_mat @ bf_vec) for cov_mat in cov_mat_list])
        beam_pattern_best = beam_pattern_best / beam_pattern_best.max()

        beam_pattern_worst = np.abs(bf_vec.conj().T @ bf_mat)
        beam_pattern_worst = beam_pattern_worst / beam_pattern_worst.max()

        # plot the array resolution
        corr = np.abs(bf_mat.conj().T @ bf_mat)

        plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1])
        ax1 = plt.subplot(gs[0], polar=True)
        ax2 = plt.subplot(gs[1], polar=False)

        # ax1.plot(doa_list, np.abs(corr[len(corr) // 2]), label="beam pattern")
        # ax1.plot(doa_list, beam_pattern_best)
        ax1.plot(doa_list, beam_pattern_worst, label="beam pattern")
        ax1.set_title(
            f"array resolution: freq= {freq_design / 1000:0.1f} KHz, ker-duration: {int(1000 * kernel_duration)} ms")
        ax1.grid(True)

        selected_indices = np.arange(0, len(corr), len(corr) // 4)
        selected_doa = doa_list[selected_indices]
        ax2.plot(doa_list / np.pi * 180, corr[selected_indices, :].T)
        ax2.legend([f"DoA: {int(rad / np.pi * 180)}" for rad in selected_doa])
        ax2.set_xlabel("DoA")
        ax2.set_ylabel("array resolution")
        ax2.grid(True)

        if SAVE_PLOTS:
            plt.savefig(filename)
        else:
            plt.draw()

    if not SAVE_PLOTS:
        plt.show()


def array_resolution_wideband():
    """
    this function computes the array resolution curve at various angle via direct numerical simulation.
    """
    # find the directory for this file
    root = os.path.join(Path(__file__).resolve().parent, "array_resolution_wideband")

    if not os.path.exists(root):
        os.mkdir(root)

    # build a geometry
    radius = 4.5e-2
    num_mic = 7
    fs = 48_000

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 10e-3

    # minimum frequency for which STHT can works
    f_min = 10 / kernel_duration

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 0.4
    bandwidth = 1000
    center_freq_vec = [1000, 2000, 3600, 4000, 8000]

    for center_freq in tqdm(center_freq_vec):
        print(f"\n\nplotting beam pattern for wideband signal of bandwidth: {center_freq}...")

        # frequency range of the array
        # freq_range = [f_min, f_min + bandwidth]
        freq_range = [center_freq - bandwidth / 2, center_freq + bandwidth / 2]
        beamf = Beamformer(geometry=geometry, kernel_duration=kernel_duration, freq_range=freq_range, fs=fs)

        # build a filter for the array
        # butterworth filter parameters
        N = 2
        b, a = butter(N, freq_range, btype='pass', analog=False, output='ba', fs=fs)

        filename = os.path.join(root, f"array_resolution_wideband_fc={center_freq}.pdf")

        time_temp = np.arange(0, duration, step=1 / fs)
        noise = np.random.randn(len(time_temp))

        sig_temp = lfilter(b, a, noise)

        # 2. use an angular grid
        num_grid = 16 * num_mic + 1
        doa_list = np.linspace(-np.pi, np.pi, num_grid)

        bf_mat, _ = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list)

        # plot the array resolution
        corr = np.abs(bf_mat.conj().T @ bf_mat)

        plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1])
        ax1 = plt.subplot(gs[0], polar=True)
        ax2 = plt.subplot(gs[1], polar=False)

        ax1.plot(doa_list, np.abs(corr[len(corr) // 2]), label="beam pattern")
        ax1.set_title(
            fr"array resolution: $f_c$: {center_freq / 1000:0.1f} KHz, B: {bandwidth / 1000:0.1f} KHz, ker-duration: {int(1000 * kernel_duration)} ms")
        ax1.grid(True)

        selected_indices = np.arange(0, len(corr), len(corr) // 4)
        selected_doa = doa_list[selected_indices]
        ax2.plot(doa_list / np.pi * 180, corr[selected_indices, :].T)
        ax2.legend([f"DoA: {int(rad / np.pi * 180)}" for rad in selected_doa])
        ax2.set_xlabel("DoA")
        ax2.set_ylabel("array resolution")
        ax2.grid(True)

        if SAVE_PLOTS:
            plt.savefig(filename)
        else:
            plt.draw()

    if not SAVE_PLOTS:
        plt.show()


def main():
    # array_resolution_sin()
    array_resolution_wideband()


if __name__ == '__main__':
    main()
