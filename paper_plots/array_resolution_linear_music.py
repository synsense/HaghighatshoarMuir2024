# ----------------------------------------------------------------------------------------------------------------------
# This module plots the array resolution for the MUSIC algorithm.
#
# NOTE: This simulation is specifically for review1 and addresses the linear array geometry rather than the circular one used
# in the first submission.
#
# NOTE: Since in linear array, we have double-sided beam, namely, beam pattern is circularly symmetric around array axis,
# we need to plot only the projection of the bema on zx plane for example.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 17.07.2024
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np

from micloc.array_geometry import LinearArray
from micloc.music_beamformer import MUSIC
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import os
from pathlib import Path
from tqdm import tqdm

from array_resolution_linear_snn import array_resolution_wideband


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



def plot_beampattern(doa_list, beam1, beam2, beam3, title, filename, geometry):
    mm = 1 / 25.4

    plt.figure(figsize=[35 * mm, 35 * mm])
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0], polar=True)
    # ax2 = plt.subplot(gs[1], polar=False)

    ax1.plot(doa_list, np.abs(beam1), label="beam pattern")
    ax1.plot(doa_list, np.abs(beam2), label="beam pattern")
    ax1.plot(doa_list, np.abs(beam3), label="beam pattern")
    ax1.set_title(title)
    ax1.grid(True)
    EPS = 0.00001
    ax1.set_xticks(np.arange(0 / 180 * np.pi, 180 / 180 * np.pi, (60-EPS) / 180 * np.pi))
    ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels([])
    ax1.set_thetamax(180)

    ax1.plot(geometry.theta_vec, geometry.r_vec / geometry.radius, "k.")

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
    root = os.path.join(
        Path(__file__).resolve().parent, "array_resolution_linear_music_sin"
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

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 0.4
    num_fft_bin = 2048
    freq_design_vec = [1000, 2000, 3600, 4000, 8000]

    # use an angular grid
    num_grid = 64 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_grid) + np.pi

    for freq_design in tqdm(freq_design_vec):
        print(f"\n\nplotting beam pattern for freq: {freq_design}...")

        filename = os.path.join(root, f"array_resolution_sin_freq={freq_design}.pdf")

        # frequency range and numebr of active frequencies
        freq_range = [0.8 * freq_design, 1.2 * freq_design]
        num_active_freq = 1

        beamf = MUSIC(
            geometry=geometry,
            freq_range=freq_range,
            doa_list=doa_list,
            frame_duration=duration,
            fs=fs,
        )

        time_temp = np.arange(0, duration, step=1 / fs)
        sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

        ## compute two beampatterns
        ang_pow_spec1 = beamf.apply_to_template(
            template=[time_temp, sig_temp, 0],
            num_active_freq=num_active_freq,
            duration_overlap=0.0,
            num_fft_bin=num_fft_bin,
            snr_db=1000,
        )

        # accumulate all angular power spectrum in tim
        beam_pattern1 = ang_pow_spec1.mean(0)
        beam_pattern1 = beam_pattern1 / beam_pattern1.max()

        ang_pow_spec2 = beamf.apply_to_template(
            template=[time_temp, sig_temp, np.pi / 2],
            num_active_freq=num_active_freq,
            duration_overlap=0.0,
            num_fft_bin=num_fft_bin,
            snr_db=1000,
        )

        # accumulate all angular power spectrum in tim
        beam_pattern2 = ang_pow_spec2.mean(0)
        beam_pattern2 = beam_pattern2 / beam_pattern2.max()

        ang_pow_spec3 = beamf.apply_to_template(
            template=[time_temp, sig_temp, 2*np.pi / 3],
            num_active_freq=num_active_freq,
            duration_overlap=0.0,
            num_fft_bin=num_fft_bin,
            snr_db=1000,
        )

        # accumulate all angular power spectrum in tim
        beam_pattern3 = ang_pow_spec3.mean(0)
        beam_pattern3 = beam_pattern3 / beam_pattern3.max()

        plot_beampattern(
            doa_list,
            beam_pattern1,
            beam_pattern2,
            beam_pattern3,
            f"$F= {freq_design / 1000:0.0f}$ kHz",
            filename,
            geometry,
        )

        # plt.figure()
        # gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1])
        # ax1 = plt.subplot(gs[0], polar=True)
        # ax2 = plt.subplot(gs[1], polar=False)

        # ax1.plot(doa_list, beam_pattern1, label="beam pattern")
        # ax1.plot(doa_list, beam_pattern2, label="beam pattern")
        # ax1.set_title(
        #     f"array resolution: freq= {freq_design / 1000:0.1f} KHz")
        # ax1.grid(True)

        # ax2.plot(doa_list / np.pi * 180, beam_pattern1)
        # ax2.plot(doa_list / np.pi * 180, beam_pattern2)
        # ax2.legend(["DoA: 0", "DoA: 90"])
        # ax2.set_xlabel("DoA")
        # ax2.set_ylabel("array resolution")
        # ax2.grid(True)

        # if SAVE_PLOTS:
        #     plt.savefig(filename)
        # else:
        #     plt.draw()

    if not SAVE_PLOTS:
        plt.show()


def main():
    array_resolution_sin()


if __name__ == "__main__":
    main()
