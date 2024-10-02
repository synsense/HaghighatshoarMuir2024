# ----------------------------------------------------------------------------------------------------------------------
# This module plots the array resolution for the Hilbert-transform based localization algorithm.
# This module obtains array beam pattern for the SNN version.
#
# NOTE: this file was added for review1 to address checking another geometry for multi-mic array.
# NOTE: in this file, we address a linear array geometry rather than the circular one used for the previous submission.
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 08.07.2024
# ----------------------------------------------------------------------------------------------------------------------
from cmath import phase
import numpy as np

from micloc.array_geometry import LinearArray
from micloc.snn_beamformer import SNNBeamformer
from micloc.utils import Envelope
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
from scipy.signal import lfilter, butter
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


def plot_beampattern(doa_list, corr, title, filename, geometry):
    mm = 1 / 25.4

    plt.figure(figsize=[35 * mm, 35 * mm])
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0], polar=True)
    ax1.set_thetamax(180)
    # ax2 = plt.subplot(gs[1], polar=False)

    # selected DoAs to be plotted
    selected_doa = [0, np.pi/2, 2*np.pi/3]

    # find the corresponding indices
    selected_indices = []
    for doa in selected_doa:
        index = np.argmin(np.abs(doa_list - doa))
        selected_indices.append(index)
    
    for index in selected_indices:
        ax1.plot(doa_list, np.abs(corr[index]), label="beam pattern")
    
    ax1.set_title(title)
    ax1.grid(True)
    EPS = 0.0001
    ax1.set_xticks(np.arange(0/180*np.pi, 180 / 180 * np.pi, (60-EPS)/ 180 * np.pi))
    ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels([])

    ax1.plot(geometry.theta_vec, geometry.r_vec / geometry.radius, "k.")

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
    """
    # find the directory for this file
    root = os.path.join(
        Path(__file__).resolve().parent, "array_resolution_linear_snn_sin"
    )

    if not os.path.exists(root):
        os.mkdir(root)

    # build a geometry
    radius = 4.5e-2
    whole_span = 2 * radius
    num_mic = 7
    spacing = whole_span / num_mic
    fs = 48_000

    # geometry = CenterCircularArray(radius=radius, num_mic=num_mic)
    geometry = LinearArray(
        spacing=spacing,
        num_mic=num_mic,
        radius=radius,
    )

    # build the corresponding beamformer
    kernel_duration = 10e-3
    freq_range = [500, 8500]

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 0.6
    freq_design_vec = [1000, 2000, 3600, 4000, 8000]

    for freq_design in tqdm(freq_design_vec):
        # design a specific beamformer
        freq_range = [freq_design / 2, 2 * freq_design]
        tau_mem = 1 / (2 * np.pi * freq_design)
        tau_syn = 1 / (2 * np.pi * freq_design)
        tau_vec = [tau_mem, tau_syn]

        # bipolar spikes
        bipolar_spikes = False
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
            root, f"array_resolution_snn_sin_freq={freq_design}.pdf"
        )

        time_temp = np.arange(0, duration, step=1 / fs)

        # NOTE: a little bit of phase jitter was added to sinusoids to make the zero-crossing less sensitive to the sampling rate
        # this is quite valid in practice since in reality we cannot have pure tunes for which the zero-crossings may be badly aligned
        # at low sampling rate
        # One can see that with this little change, the beam pattern becomes quite smooth compared to when there is almost no jitter.
        EPS = 0.01
        freq_inst = freq_design * (1 + EPS*np.random.randn(len(time_temp)))
        dt = 1/fs
        phase_inst = 2*np.pi*np.cumsum(freq_inst) * dt
        sig_temp = np.sin(phase_inst)
        # sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

        # 2. use an angular grid
        num_grid = 64 * num_mic + 1
        doa_list = np.linspace(0, np.pi, num_grid)

        bf_mat = beamf.design_from_template(
            template=(time_temp, sig_temp), doa_list=doa_list
        )

        # obtain the complex version of the beamforming
        bf_mat_comp = bf_mat[:num_mic, :] + 1j * bf_mat[num_mic:, :]

        # plot the array resolution
        corr = np.abs(bf_mat_comp.conj().T @ bf_mat_comp)

        plot_beampattern(
            doa_list, corr, f"$F= {freq_design / 1000:0.0f}$ kHz", filename, geometry
        )

    if not SAVE_PLOTS:
        plt.show()


def array_resolution_wideband():
    """
    this function computes the array resolution curve at various angle via direct numerical simulation.
    """
    # find the directory for this file
    root = os.path.join(
        Path(__file__).resolve().parent, "array_resolution_linear_snn_wideband"
    )

    if not os.path.exists(root):
        os.mkdir(root)

    # build a geometry
    radius = 4.5e-2
    whole_span = 2 * radius
    num_mic = 7
    spacing = whole_span / num_mic
    fs = 48_000

    # geometry = CenterCircularArray(radius=radius, num_mic=num_mic)
    geometry = LinearArray(
        spacing=spacing,
        num_mic=num_mic,
        radius=radius,
    )

    # build the corresponding beamformer
    kernel_duration = 10e-3

    # minimum frequency for which STHT can works
    f_min = 10 / kernel_duration

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 0.6
    bandwidth = 1000
    center_freq_vec = [1000, 2000, 3600, 4000, 8000]

    for center_freq in tqdm(center_freq_vec):
        print(
            f"\n\nplotting beam pattern for wideband signal of bandwidth: {center_freq}..."
        )

        # design a specific beamformer
        freq_range = [center_freq / 2, 2 * center_freq]
        tau_mem = 1 / (2 * np.pi * center_freq)
        tau_syn = 1 / (2 * np.pi * center_freq)
        tau_vec = [tau_mem, tau_syn]

        # frequency range of the array
        # freq_range = [f_min, f_min + bandwidth]
        freq_range = [center_freq - bandwidth / 2, center_freq + bandwidth / 2]

        # build the beamformer for the specific scenario
        bipolar_spikes = False

        beamf = SNNBeamformer(
            geometry=geometry,
            kernel_duration=kernel_duration,
            freq_range=freq_range,
            tau_vec=tau_vec,
            bipolar_spikes=bipolar_spikes,
            fs=fs,
        )

        # build a filter for the array
        # butterworth filter parameters
        N = 2
        b, a = butter(N, freq_range, btype="pass", analog=False, output="ba", fs=fs)

        filename = os.path.join(
            root, f"array_resolution_snn_wideband_fc={center_freq}.pdf"
        )

        time_temp = np.arange(0, duration, step=1 / fs)
        noise = np.random.randn(len(time_temp))

        sig_temp = lfilter(b, a, noise)

        # 2. use an angular grid
        num_grid = 32 * num_mic + 1
        doa_list = np.linspace(-np.pi, np.pi, num_grid) + np.pi

        bf_mat = beamf.design_from_template(
            template=(time_temp, sig_temp), doa_list=doa_list
        )

        # obtain the complex version of the beamforming matrix
        bf_mat_comp = bf_mat[:num_mic, :] + 1j * bf_mat[num_mic:, :]

        # plot the array resolution
        # corr = np.abs(bf_mat.conj().T @ bf_mat)
        corr = np.abs(bf_mat_comp.conj().T @ bf_mat_comp)

        plot_beampattern(
            doa_list,
            corr,
            f"$F_c= {center_freq / 1000:0.0f}$ kHz, $B={bandwidth / 1000:0.0f}$ kHz",
            filename,
            geometry,
        )

    if not SAVE_PLOTS:
        plt.show()


def main():
    array_resolution_sin()
    # plot_beampattern()
    array_resolution_wideband()


if __name__ == "__main__":
    main()
