# ----------------------------------------------------------------------------------------------------------------------
# This module plots the array resolution for the Hilbert-transform based localization algorithm.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 30.08.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np

from micloc.array_geometry import CenterCircularArray
from micloc.beamformer import Beamformer
from micloc.utils import Envelope
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
from scipy.signal import lfilter, butter
import os
from pathlib import Path
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


def array_resolution_sin():
    """
    this function computes the array resolution curve at various angle via direct numerical simulation.
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
    beamf = Beamformer(geometry=geometry, kernel_duration=kernel_duration, fs=fs)

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 0.4
    freq_design_vec = [1000, 2000, 4000, 8000]

    for freq_design in tqdm(freq_design_vec):
        print(f"\n\nplotting beam pattern for freq: {freq_design}...")

        filename = os.path.join(root, f"array_resolution_sin_freq={freq_design}.pgf")

        time_temp = np.arange(0, duration, step=1 / fs)
        sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

        # 2. use an angular grid
        num_grid = 16 * num_mic + 1
        doa_list = np.linspace(-np.pi, np.pi, num_grid)

        bf_mat = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list)

        # plot the array resolution
        corr = np.abs(bf_mat.conj().T @ bf_mat)

        plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1])
        ax1 = plt.subplot(gs[0], polar=True)
        ax2 = plt.subplot(gs[1], polar=False)

        ax1.plot(doa_list, np.abs(corr[len(corr) // 2]), label="beam pattern")
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
    beamf = Beamformer(geometry=geometry, kernel_duration=kernel_duration, fs=fs)

    # minimum frequency for which STHT can works
    f_min = 10 / kernel_duration

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 0.4
    bandwidth = 1000
    center_freq_vec = [1000, 2000, 4000, 8000]

    for center_freq in tqdm(center_freq_vec):
        print(f"\n\nplotting beam pattern for wideband signal of bandwidth: {center_freq}...")

        # frequency range of the array
        # freq_range = [f_min, f_min + bandwidth]
        freq_range = [center_freq - bandwidth/2, center_freq + bandwidth/2]

        # build a filter for the array
        # butterworth filter parameters
        N = 2
        b, a = butter(N, freq_range, btype='pass', analog=False, output='ba', fs=fs)

        filename = os.path.join(root, f"array_resolution_wideband_fc={center_freq}.pgf")

        time_temp = np.arange(0, duration, step=1 / fs)
        noise = np.random.randn(len(time_temp))

        sig_temp = lfilter(b, a, noise)

        # 2. use an angular grid
        num_grid = 16 * num_mic + 1
        doa_list = np.linspace(-np.pi, np.pi, num_grid)

        bf_mat = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list)

        # plot the array resolution
        corr = np.abs(bf_mat.conj().T @ bf_mat)

        plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1])
        ax1 = plt.subplot(gs[0], polar=True)
        ax2 = plt.subplot(gs[1], polar=False)

        ax1.plot(doa_list, np.abs(corr[len(corr) // 2]), label="beam pattern")
        ax1.set_title(
            fr"array resolution: $f_c$: {center_freq/1000:0.1f} KHz, B: {bandwidth / 1000:0.1f} KHz, ker-duration: {int(1000 * kernel_duration)} ms")
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
    array_resolution_sin()
    array_resolution_wideband()


if __name__ == '__main__':
    main()
