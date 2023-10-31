# ----------------------------------------------------------------------------------------------------------------------
# This module investigates the localization performance for SNN when there are several targets.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 31.10.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np

from micloc.array_geometry import ArrayGeometry, CenterCircularArray
from micloc.snn_beamformer import SNNBeamformer

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


def signal_multiple_targets(geometry: ArrayGeometry, time_temp: np.ndarray, sig_temp: np.ndarray,
                            doa_timeseries_targets: List[List[float]], power_timeseries_targets: List[List[float]]):
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
        raise ValueError("time vector and input signal should have the same dimensions!")

    doa_timeseries_targets = np.asarray(doa_timeseries_targets)
    power_timeseries_targets = np.asarray(power_timeseries_targets)

    if doa_timeseries_targets.ndim == 1:
        doa_timeseries_targets = doa_timeseries_targets.reshape(-1, 1)

    if power_timeseries_targets.ndim == 1:
        power_timeseries_targets = power_timeseries_targets.reshape(-1, 1)

    T_doa, num_targets_doa = doa_timeseries_targets.shape
    T_power, num_targets_power = power_timeseries_targets.shape

    if num_targets_doa != num_targets_power:
        raise ValueError("number of targets should be the same in doa and power vector!")

    if T_doa != T_power or T_doa != T:
        raise ValueError("input signal, doa, and power vectors should have the same dimension!")

    num_targets = num_targets_doa

    sig_in = 0

    for idx in range(num_targets):
        # compute the delay at doas: output `T x num_mic`
        delay_vec = np.asarray([geometry.delays(doa, normalized=False) for doa in doa_timeseries_targets[:, idx]])

        # compute time-delayed vec: output `T x num_mic`
        time_vec = time_temp.reshape(-1, 1) + delay_vec

        # interpolate the signal -> output `T x num_mic`
        sig_target = np.interp(time_vec.ravel(), time_temp, sig_temp).reshape(*time_vec.shape)

        # apply the power scaling: output `T x num_mic`
        sig_target = power_timeseries_targets[:, idx].reshape(-1, 1) * sig_target

        sig_in += sig_target

    return sig_in


# ===========================================================================
#                                simulations
# ===========================================================================

def snn_multiple_targets_sin():
    """
    this function computes the localization performance when there are several targets.
    """
    # find the directory for this file
    root = os.path.join(Path(__file__).resolve().parent, "snn_multiple_targets_sin")

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

    # how many targets: their doa and power level
    # NOTE: here for simplicity we work with fixed targets
    doa_targets = np.asarray([-np.pi / 3, np.pi / 3])
    power_targets = np.asarray([1, 1])

    for freq_design in tqdm(freq_design_vec):
        # design a specific beamformer
        freq_range = np.array([freq_design / 2, 2 * freq_design])
        tau_mem = 1 / (2 * np.pi * freq_design)
        tau_syn = 1 / (2 * np.pi * freq_design)
        tau_vec = np.asarray([tau_mem, tau_syn])

        # bipolar spikes
        bipolar_spikes = True
        beamf = SNNBeamformer(
            geometry=geometry,
            kernel_duration=kernel_duration,
            freq_range=freq_range,
            tau_vec=tau_vec,
            bipolar_spikes=bipolar_spikes,
            fs=fs,
        )

        print(f"\n\nplotting beam pattern for freq: {freq_design}...")

        filename = os.path.join(
            root, f"multiple_targets_snn_sin_freq={freq_design}.pdf"
        )

        time_temp = np.arange(0, duration, step=1 / fs)
        sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

        # 2. use an angular grid
        num_grid = 32 * num_mic + 1
        doa_list = np.linspace(-np.pi, np.pi, num_grid)

        bf_mat = beamf.design_from_template(
            template=(time_temp, sig_temp), doa_list=doa_list
        )

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
        sig_bf = beamf.apply_to_signal(bf_mat=bf_mat, sig_in_vec=(time_temp, sig_in))

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

        plot_beampattern(doa_list, power_bf, f"$F= {freq_design / 1000:0.0f}$ kHz", filename)

    if not SAVE_PLOTS:
        plt.show()


def snn_multiple_targets_wideband():
    """
    this function evaluates the performance of SNN localization for the wideband scenario.
    """
    # find the directory for this file
    root = os.path.join(
        Path(__file__).resolve().parent, "snn_multiple_targets_wideband"
    )

    if not os.path.exists(root):
        os.mkdir(root)

    # build a geometry
    radius = 4.5e-2
    num_mic = 7
    fs = 48_000

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 10e-3

    # minimum frequency for which STHT can work
    f_min = 10 / kernel_duration

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 0.4
    bandwidth = 1000
    center_freq_vec = [1000, 2000, 3600, 4000, 8000]

    # DoAs of targets
    doa_targets = np.array([np.pi / 3, -np.pi / 3])
    power_targets = np.array([1, 1])

    for center_freq in tqdm(center_freq_vec):
        print(
            f"\n\nplotting beam pattern for wideband signal of bandwidth: {center_freq}..."
        )

        # design a specific beamformer
        freq_range = [center_freq / 2, 2 * center_freq]
        tau_mem = 1 / (2 * np.pi * center_freq)
        tau_syn = 1 / (2 * np.pi * center_freq)
        tau_vec = np.asarray([tau_mem, tau_syn])

        # frequency range of the array
        # freq_range = [f_min, f_min + bandwidth]
        freq_range = np.asarray([center_freq - bandwidth / 2, center_freq + bandwidth / 2])

        ## build a temlate signal for the array
        # build a filter for the array
        # butterworth filter parameters
        N = 2
        b, a = butter(N, freq_range, btype="pass", analog=False, output="ba", fs=fs)

        time_temp = np.arange(0, duration, step=1 / fs)
        noise = np.random.randn(len(time_temp))

        sig_temp = lfilter(b, a, noise)

        # use an angular grid
        num_grid = 32 * num_mic + 1
        doa_list = np.linspace(-np.pi, np.pi, num_grid)

        # bipolar spikes
        bipolar_spikes = True
        beamf = SNNBeamformer(
            geometry=geometry,
            kernel_duration=kernel_duration,
            freq_range=freq_range,
            tau_vec=tau_vec,
            bipolar_spikes=bipolar_spikes,
            fs=fs,
        )

        # build the beamformer for the specific scenario
        bf_mat = beamf.design_from_template(
            template=(time_temp, sig_temp), doa_list=doa_list
        )

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
        sig_bf = beamf.apply_to_signal(bf_mat=bf_mat, sig_in_vec=(time_temp, sig_in))

        # compute the power along different DoAs in the grid
        power_bf = np.mean(np.abs(sig_bf) ** 2, axis=0)

        # apply normalization
        power_bf = power_bf / power_bf.max()

        print(f"\n\nplotting beam pattern for freq-range {freq_range}...")

        # plot
        # plt.figure()
        # plt.plot(doa_list/np.pi*180, power_bf, label="power profile")
        # for idx, doa in enumerate(doa_targets):
        #     doa_deg = doa/np.pi * 180
        #     plt.axvline(x=doa_deg, color="r", label=f"doa target#{idx}: {doa_deg:0.1f} deg")
        # plt.grid(True)
        # plt.xlabel("DoA")
        # plt.ylabel("power profile after beamforming")
        # plt.title(f"center-freq: {center_freq/1000: 0.1f} KHz, bw: {bandwidth/1000:0.2f} KHz, num-mic: {num_mic}")
        # plt.legend()
        # plt.draw()

        filename = os.path.join(
            root, f"multiple_targets_snn_wideband_center_freq={center_freq}.pdf"
        )

        plot_beampattern(doa_list, power_bf, f"$F_c= {center_freq / 1000:0.1f}$ kHz, $B={bandwidth / 1000:0.1f}$ kHz",
                         filename)

    if not SAVE_PLOTS:
        plt.show()


def main():
    # snn_multiple_targets_sin()
    snn_multiple_targets_wideband()


if __name__ == "__main__":
    main()
