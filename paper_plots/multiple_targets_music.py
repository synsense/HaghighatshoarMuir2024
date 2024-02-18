# ----------------------------------------------------------------------------------------------------------------------
# This module investigates the localization performance for MUSIC when there are several targets.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 25.01.2024
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np

from micloc.array_geometry import ArrayGeometry, CenterCircularArray
from micloc.music_beamformer import MUSIC

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
from scipy.signal import lfilter, butter
import os
from pathlib import Path
from tqdm import tqdm

from typing import List


def use_latex():
    matplotlib.use("pdf")
    matplotlib.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
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


def plot_beampattern(doa_list, power_profile, title, filename):
    mm = 1 / 25.4

    plt.figure(figsize=[35 * mm, 35 * mm])
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0], polar=True)
    # ax2 = plt.subplot(gs[1], polar=False)

    ax1.plot(doa_list, power_profile, label="beam pattern")
    ax1.set_title(title)
    ax1.grid(True)
    ax1.set_xticks(np.arange(0 / 180 * np.pi, 360 / 180 * np.pi, 60 / 180 * np.pi))
    ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels([])

    if SAVE_PLOTS:
        plt.savefig(filename, bbox_inches="tight", transparent=True)
    else:
        plt.draw()


def signal_multiple_targets(
    geometry: ArrayGeometry,
    time_temp: np.ndarray,
    sig_temp: np.ndarray,
    doa_timeseries_targets: List[List[float]],
    power_timeseries_targets: List[List[float]],
):
    """this function computes the input signal

    Args:
        geometry(ArrayGeometry): geometry of the array.
        time_temp (np.ndarray): time in template signal.
        sig_temp (np.ndarray): signal in template signal.
        doa_timeseries_targets (List[List[float]]): a list containing DoAs of target/targets.
        power_timeseries_targets (List[List[float]]): a list containing powers of target/targets.
    """

    # some sanity check
    T = len(time_temp)
    if T != len(sig_temp):
        raise ValueError(
            "time vector and input signal should have the same dimensions!"
        )

    doa_timeseries_targets = np.asarray(doa_timeseries_targets)
    power_timeseries_targets = np.asarray(power_timeseries_targets)

    if doa_timeseries_targets.ndim == 1:
        doa_timeseries_targets = doa_timeseries_targets.reshape(-1, 1)

    if power_timeseries_targets.ndim == 1:
        power_timeseries_targets = power_timeseries_targets.reshape(-1, 1)

    T_doa, num_targets_doa = doa_timeseries_targets.shape
    T_power, num_targets_power = power_timeseries_targets.shape

    if num_targets_doa != num_targets_power:
        raise ValueError(
            "number of targets should be the same in doa and power vector!"
        )

    if T_doa != T_power or T_doa != T:
        raise ValueError(
            "input signal, doa, and power vectors should have the same dimension!"
        )

    num_targets = num_targets_doa

    sig_in = 0

    for idx in range(num_targets):
        # compute the delay at doas: output `T x num_mic`
        delay_vec = np.asarray(
            [
                geometry.delays(doa, normalized=False)
                for doa in doa_timeseries_targets[:, idx]
            ]
        )

        # compute time-delayed vec: output `T x num_mic`
        time_vec = time_temp.reshape(-1, 1) - delay_vec

        # interpolate the signal -> output `T x num_mic`
        sig_target = np.interp(time_vec.ravel(), time_temp, sig_temp).reshape(
            *time_vec.shape
        )

        # apply the power scaling: output `T x num_mic`
        sig_target = power_timeseries_targets[:, idx].reshape(-1, 1) * sig_target

        sig_in += sig_target

    return sig_in


# ===========================================================================
#                               simulations
# ===========================================================================


def music_multiple_targets_sin():
    """
    this function computes the localization performance when there are several targets.
    NOTE: we assume that the spectrum of the signal received from each target is a sinusoid.
    """
    # find the directory for this file
    root = os.path.join(Path(__file__).resolve().parent, "music_multiple_targets_sin")

    if not os.path.exists(root):
        os.mkdir(root)

    # build a geometry
    radius = 4.5e-2
    num_mic = 7
    fs = 48_000

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # number of grid elements
    num_gird = 32 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_gird)

    # duration of signal and sample frequencies
    duration = 0.4
    num_fft_bin = 2048
    freq_design_vec = [1000, 2000, 3600, 4000, 8000]

    # how many targets: their doa and power level
    # NOTE: here for simplicity we work with fixed targets
    doa_targets = np.asarray([-np.pi / 3, np.pi / 3])
    power_targets = np.asarray([1, 1])

    for freq_design in tqdm(freq_design_vec):
        # design a specific beamformer
        freq_range = np.asarray([freq_design / 2, 2 * freq_design])

        # number of active frequencies
        music = MUSIC(
            geometry=geometry,
            freq_range=freq_range,
            doa_list=doa_list,
            frame_duration=duration,
            fs=fs,
        )

        print(f"\n\nplotting beam pattern for freq: {freq_design}...")

        filename = os.path.join(
            root, f"multiple_targets_music_sin_freq={freq_design}.pdf"
        )

        time_temp = np.arange(0, duration, step=1 / fs)
        sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

        # compute the input signal for the targets
        T = len(sig_temp)
        doa_timeseries_targets = np.ones((T, 1)) * doa_targets.reshape(1, -1)
        power_timeseries_targets = np.ones((T, 1)) * power_targets.reshape(1, -1)

        sig_in = signal_multiple_targets(
            geometry=geometry,
            time_temp=time_temp,
            sig_temp=sig_temp,
            doa_timeseries_targets=doa_timeseries_targets,
            power_timeseries_targets=power_timeseries_targets,
        )

        # apply all SNN signal processing chain to input signal
        num_active_freq = 1  # just choose the strongest one
        duration_overlap = 0.0  # oneshot MUSIC
        sig_bf = music.apply_to_signal(
            sig_in=sig_in,
            num_active_freq=num_active_freq,
            duration_overlap=duration_overlap,
            num_fft_bin=num_fft_bin,
        )

        # compute the power along different DoAs in the grid
        power_bf = np.mean(np.abs(sig_bf) ** 2, axis=0)

        # apply normalization
        power_bf = power_bf / power_bf.max()

        # plot
        # plt.figure()
        # plt.plot(doa_list/np.pi*180, power_bf, label="power profile")
        # for idx, doa in enumerate(doa_targets):
        #     doa_deg = doa/np.pi * 180
        #     plt.axvline(x=doa_deg, color="r", label=f"doa target#{idx}: {doa_deg:0.1f} deg")
        # plt.grid(True)
        # plt.xlabel("DoA")
        # plt.ylabel("power profile after beamforming")
        # plt.title(f"freq: {freq_design/1000: 0.1f} KHz, num-mic: {num_mic}")
        # plt.legend()
        # plt.draw()

        plot_beampattern(
            doa_list, power_bf, f"$F= {freq_design / 1000:0.0f}$ kHz", filename
        )

    if not SAVE_PLOTS:
        plt.show()


def music_multiple_targets_wideband():
    """
    this function computes the localization performance when there are several targets.
    NOTE: we assume that the spectrum of the signal received from each target is a colored Gaussian noise.
    """
    # find the directory for this file
    root = os.path.join(Path(__file__).resolve().parent, "music_multiple_targets_sin")

    if not os.path.exists(root):
        os.mkdir(root)

    # build a geometry
    radius = 4.5e-2
    num_mic = 7
    fs = 48_000

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # number of grid elements
    num_gird = 32 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_gird)

    # duration of signal and sample frequencies
    duration = 0.4
    num_fft_bin = 2048
    freq_design_vec = [1000, 2000, 3600, 4000, 8000]
    bandwidth = 1000

    # how many targets: their doa and power level
    # NOTE: here for simplicity we work with fixed targets
    doa_targets = np.asarray([-np.pi / 3, np.pi / 3])
    power_targets = np.asarray([1, 1])

    for freq_design in tqdm(freq_design_vec):
        # design a specific beamformer
        freq_range = np.asarray(
            [freq_design - bandwidth / 2, freq_design + bandwidth / 2]
        )

        # number of active frequencies
        music = MUSIC(
            geometry=geometry,
            freq_range=freq_range,
            doa_list=doa_list,
            frame_duration=duration,
            fs=fs,
        )

        print(f"\n\nplotting beam pattern for freq: {freq_design}...")

        filename = os.path.join(
            root, f"multiple_targets_music_sin_freq={freq_design}.pdf"
        )

        ## build a temlate signal for the array

        time_temp = np.arange(0, duration, step=1 / fs)

        # build a filter for the array
        # butterworth filter parameters
        N = 2
        b, a = butter(N, freq_range, btype="pass", analog=False, output="ba", fs=fs)

        time_temp = np.arange(0, duration, step=1 / fs)
        noise = np.random.randn(len(time_temp))

        sig_temp = lfilter(b, a, noise)

        # compute the input signal for the targets
        T = len(sig_temp)
        doa_timeseries_targets = np.ones((T, 1)) * doa_targets.reshape(1, -1)
        power_timeseries_targets = np.ones((T, 1)) * power_targets.reshape(1, -1)

        sig_in = signal_multiple_targets(
            geometry=geometry,
            time_temp=time_temp,
            sig_temp=sig_temp,
            doa_timeseries_targets=doa_timeseries_targets,
            power_timeseries_targets=power_timeseries_targets,
        )

        # apply all SNN signal processing chain to input signal
        num_active_freq = 2  # just choose the strongest one or we may choose multiple frequencies since each target can be strong in one
        duration_overlap = duration / 4  # oneshot MUSIC
        sig_bf = music.apply_to_signal(
            sig_in=sig_in,
            num_active_freq=num_active_freq,
            duration_overlap=duration_overlap,
            num_fft_bin=num_fft_bin,
        )

        # compute the power along different DoAs in the grid
        power_bf = np.mean(np.abs(sig_bf) ** 2, axis=0)

        # apply normalization
        power_bf = power_bf / power_bf.max()

        # plot
        # plt.figure()
        # plt.plot(doa_list/np.pi*180, power_bf, label="power profile")
        # for idx, doa in enumerate(doa_targets):
        #     doa_deg = doa/np.pi * 180
        #     plt.axvline(x=doa_deg, color="r", label=f"doa target#{idx}: {doa_deg:0.1f} deg")
        # plt.grid(True)
        # plt.xlabel("DoA")
        # plt.ylabel("power profile after beamforming")
        # plt.title(f"freq: {freq_design/1000: 0.1f} KHz, num-mic: {num_mic}")
        # plt.legend()
        # plt.draw()

        plot_beampattern(
            doa_list, power_bf, f"$F= {freq_design / 1000:0.0f}$ kHz", filename
        )

    if not SAVE_PLOTS:
        plt.show()


def main():
    music_multiple_targets_sin()
    # music_multiple_targets_wideband()


if __name__ == "__main__":
    main()
