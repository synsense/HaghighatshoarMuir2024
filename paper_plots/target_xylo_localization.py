# ----------------------------------------------------------------------------------------------------------------------
# This module uses XyloSim SNN beamforming to localize and track real targets.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 01.11.2023
# ----------------------------------------------------------------------------------------------------------------------
from numbers import Number
import numpy as np
import cvxpy as cp
import soundfile as sf
from micloc.array_geometry import ArrayGeometry, CenterCircularArray
from micloc.xylo_snn_localiztaion import Demo
from micloc.utils import Envelope
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import os
from tqdm import tqdm
from typing import Tuple
from micloc.utils import find_peak_location


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


SAVE_PLOTS = True
num_sim = 100

if SAVE_PLOTS:
    use_latex()


def approx_decreasing(sig_in):
    """
    this module finds the best monotone approximation of a given input.
    """

    # Construct the problem.
    x = cp.Variable(len(sig_in))

    # objective = cp.Minimize(cp.sum(cp.abs(sig_in - x)))
    objective = cp.Minimize(cp.sum_squares(sig_in - x))
    constraints = [x[1:] <= x[:-1]]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    sig_est = x.value

    return sig_est


def signal_from_template(geometry: ArrayGeometry, template: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """this function builds the audio signal received from the microphone from the template signal.

    Args:
        geometry (ArrayGeometry): array geometry.
        template (Tuple[np.ndarray, np.ndarray]): template signal containing (time_temp, sig_temp, doa_temp).

    Returns:
        np.ndarray: `T x num_mic` signal received in the array.
    """

    time_temp, sig_temp, doa_temp = template

    if isinstance(doa_temp, Number):
        doa_temp = doa_temp * np.ones_like(time_temp)

    # compute the delay time-series
    delays = np.asarray([geometry.delays(doa, normalized=False) for doa in doa_temp])
    time_delays = time_temp.reshape(-1, 1) + delays

    # `T x num_mic` signal received at the input of the array
    sig_in = np.interp(time_delays.ravel(), time_temp, sig_temp).reshape(*time_delays.shape)

    return sig_in


# ===========================================================================
#                              Simulations
# ===========================================================================

def test_speech_target():
    """
    this function evaluates the localization performance for a fixed speech target.
    """

    dir_name = os.path.join(Path(__file__).resolve().parent, "xylo_fixed_target")
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # build a geometry
    radius = 4.5e-2
    num_mic = 7
    fs = 48_000

    freq_design = 2_000
    freq_range = [0.5 * freq_design, freq_design]

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 10.0e-3

    # type of spike encoding
    bipolar_spikes = True

    # list of DoAs
    num_grid = 64 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    # duration of the signal used for 
    duration = 1000e-3

    # use Demo as a dummy object for computation
    # NOTE: beamforming matrices are produced within this module
    xylosim = Demo(
        geometry=geometry,
        freq_bands=[freq_range],
        doa_list=doa_list,
        recording_duration=duration,
        kernel_duration=kernel_duration,
        bipolar_spikes=bipolar_spikes,
        xylosim_version=True,
        fs=fs,
    )

    # - sinusoid as test signal
    # NOTE: here we are using a little bit longer duration than previous SNN version to obtain better averaging
    test_duration = duration / 2

    f_min, f_max = freq_range

    freq_test = (f_min + f_max) / 2
    time_test = np.arange(0, test_duration, step=1 / fs)
    sig_test = np.sin(2 * np.pi * freq_test * time_test)

    # - chirp signal
    time_test = np.arange(0, test_duration, step=1 / fs)
    freq_inst = f_min + (f_max - f_min) * time_test / test_duration
    phase_inst = 2 * np.pi * np.cumsum(freq_inst) * 1 / fs
    sig_test = np.sin(phase_inst)

    # - Load a speech sample
    sig_test, samplefreq = sf.read('84-121123-0020.flac')
    time_test = np.arange(len(sig_test)) / samplefreq
    time_fs = np.linspace(time_test[0], time_test[-1], int(len(sig_test) / samplefreq * fs))
    sig_test = np.interp(time_fs, time_test, sig_test)

    # random DoA index
    rand_doa_index = int(np.random.rand(1)[0] * len(doa_list))
    rand_doa_index = len(doa_list) // 2  # len(doa_list) // 4
    doa_target = doa_list[rand_doa_index]

    # set to 0 for this test to get the centered picture
    doa_target = 0.0

    # build the input signal from the template
    sig_in = signal_from_template(template=(time_fs, sig_test, doa_target), geometry=geometry)

    # we work with SNR in bandwidth when only noise within the bandwidth is considered
    snr_gain_due_to_bandwidth = (fs / 2) / (f_max - f_min)

    snr_db_vec = [-10, 0, 10, 20]

    mm = 1 / 25.4

    plt.figure(figsize=[40 * mm, 40 * mm])

    filename = os.path.join(dir_name, "xylo_fixed_speech_beam.pdf")

    for ind, snr_db in enumerate(snr_db_vec):
        # modify SNR to take bandwidth into account
        snr_db_bandwidth = snr_db - 10 * np.log10(snr_gain_due_to_bandwidth)

        snr_bandwidth = 10 ** (snr_db_bandwidth / 10)

        # add noise to the test signal
        sig_pow = np.mean(sig_in ** 2)
        noise_sigma = np.sqrt(sig_pow / snr_bandwidth)

        sig_in_noisy = sig_in + noise_sigma * np.random.randn(*sig_in.shape)

        # apply spike encoding to get the spikes
        spikes_in = xylosim.spike_encoding(sig_in=sig_in_noisy)

        # process the spikes by XyloSim SNN and produce output spikes
        spikes_out = xylosim.xylo_process(spikes_in=spikes_in)

        # compute the spike rate profile as a measure of power
        power = np.mean(spikes_out, axis=0) * fs
        power /= power.max()

        plt.plot(
            doa_list / np.pi * 180, 10 * np.log10(power), label=f"snr: {snr_db} dB",
            color=f'C{ind}',
        )
        plt.xlabel("DoA (deg.)")

        plt.axvline(x=doa_list[np.argmax(power)] / np.pi * 180, color=f'C{ind}', label="target DoA", linestyle='--', )

    plt.ylabel("Normalized Power (dB)")
    plt.xticks([-180, -90, 0, 90, 180])
    # plt.ylim([0, 1.05])
    # plt.legend()
    plt.grid(False)
    # plt.title(
    #     f"Angular power spectrum after beamforming:\nfreq-range:[{f_min / 1000:0.1f}, {f_max / 1000:0.1f}] KHz, DoA-target: {doa_target * 180 / np.pi:0.2f} deg"
    # )
    plt.axvline(x=0., color="k", label="target DoA", linestyle=':')
    # plt.legend()

    if SAVE_PLOTS:
        plt.savefig(filename, bbox_inches="tight", transparent=True)
    else:
        plt.show()

    # apply statistical analysis of the precision of DoA estimation
    snr_db_vec = np.linspace(-10, 20, 11)
    # snr_db_vec = np.linspace(0,20, 5)
    angle_err = []

    # test_duration = 1000e-3
    # time_test = np.arange(0, test_duration, step=1 / fs)
    # sig_test = np.sin(2 * np.pi * freq_design * time_test)

    # # NOTE: use another signal for test since sinusoids at higher frequency can be problematic due to zero-crossing
    # # - use a chirp signal
    # f_min, f_max = freq_range
    # period = time_test[-1]
    # freq_inst = f_min + (f_max - f_min) * (time_test % period) / period
    # phase_inst = 2 * np.pi * np.cumsum(freq_inst) * 1 / fs

    # sig_test = np.sin(phase_inst)

    print("\n", " statistical analysis ".center(150, "*"), "\n")

    doa_err_vec = np.zeros((np.size(snr_db_vec), num_sim))

    for snr_ind, snr_db in tqdm(enumerate(snr_db_vec)):
        print(f"statistical analysis for snr-db: {snr_db}\n")

        # snr correction considering the snr improvement after filtering
        snr_db_target = snr_db
        snr_target = 10 ** (snr_db_target / 10)

        for sim in tqdm(range(num_sim)):
            doa_target = np.random.rand(1)[0] * 2 * np.pi

            # obtain the corresponding input signal in the input of array
            sig_in = signal_from_template(template=(time_fs, sig_test, doa_target), geometry=geometry)

            # add noise to the signal
            sig_pow = np.mean(sig_in ** 2)
            noise_sigma = np.sqrt(sig_pow / snr_target)

            sig_in_noisy = sig_in + noise_sigma * np.random.randn(*sig_in.shape)

            # apply spike encoding to get the spikes
            spikes_in = xylosim.spike_encoding(sig_in=sig_in_noisy)

            # process the spikes by XyloSim SNN and produce output spikes
            spikes_out = xylosim.xylo_process(spikes_in=spikes_in)

            # compute the spike rate profile as a measure of power
            power = np.mean(spikes_out, axis=0) * fs
            power /= power.max()

            # estimate the DoA from power
            # method 1: without any denoising
            doa_target_est = doa_list[np.argmax(power)]

            # method 2: more robust for finding the location of peaks in power vector
            win_size = num_grid//32
            win_size = 2 * (win_size//2) + 1
            doa_target_index = find_peak_location(sig_in=power, win_size=win_size)
            doa_target_est = doa_list[doa_target_index]

            doa_err = np.arcsin(np.abs(np.sin(doa_target_est - doa_target)))
            doa_err_vec[snr_ind, sim] = doa_err

        # compute MAE error
        doa_err_avg = np.mean(np.abs(doa_err))

        angle_err.append(doa_err_avg)

    # plot
    # NOTE: we find the best monotone approximation
    angle_err = np.asarray(angle_err)
    # angle_err = approx_decreasing(np.asarray(angle_err))

    plt.figure(figsize=[40 * mm, 40 * mm])

    plt.plot(snr_db_vec, angle_err / np.pi * 180)
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Angle error (deg.)")

    num_xticks = 5

    filename = os.path.join(dir_name, "xylo_speech_target_accuracy_snr.pdf")

    plt.figure(figsize=[40 * mm, 40 * mm])

    plt.boxplot(doa_err_vec.T / np.pi * 180, vert=True, labels=snr_db_vec, manage_ticks=False,
                medianprops={'linestyle': '-', 'color': 'black', 'linewidth': 0.5}, boxprops={'linewidth': 0.5},
                showcaps=False, sym='k.', flierprops={'markersize': 2}, whiskerprops={'linewidth': 0.5})
    plt.plot([0, len(snr_db_vec) + 1], [1, 1], 'k:', linewidth=0.5)
    # plt.xticks(range(1, len(doa_err_vec), 5))
    xticks = np.linspace(1, len(doa_err_vec), num_xticks)
    print(xticks)
    plt.xticks(
        xticks,
        [f'{t:0.1f}' for t in np.linspace(np.min(snr_db_vec), np.max(snr_db_vec), num_xticks)],
    )
    plt.xlabel("SNR (dB)")
    plt.ylabel("Angle error (deg.)")
    plt.ylim([-2, 20])

    print(f'SNR: {snr_db_vec}')
    print(f'Mean absolute errors: {np.mean(doa_err_vec, axis=1) * 180 / np.pi}')

    if not SAVE_PLOTS:
        plt.draw()
    else:
        plt.savefig(filename, bbox_inches="tight", transparent=True)

    if not SAVE_PLOTS:
        plt.show()


def test_noisy_target():
    """
    this function evaluates the localization performance for a fixed  and non-moving target.
    """

    dir_name = os.path.join(Path(__file__).resolve().parent, "xylo_fixed_target")
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # build a geometry
    radius = 4.5e-2
    num_mic = 7
    fs = 48_000

    freq_design = 2_000
    freq_range = [0.5 * freq_design, freq_design]

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 10.0e-3

    # type of spike encoding
    bipolar_spikes = True

    # list of DoAs
    num_grid = 64 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    # duration of the signal used for
    duration = 1000e-3

    # use Demo as a dummy object for computation
    # NOTE: beamforming matrices are produced within this module
    # we are using only a single frequency band for localization
    xylosim = Demo(
        geometry=geometry,
        freq_bands=[freq_range],
        doa_list=doa_list,
        recording_duration=duration,
        kernel_duration=kernel_duration,
        bipolar_spikes=bipolar_spikes,
        xylosim_version=True,
        fs=fs,
    )

    # test scenario
    duration_test = duration / 2

    time_test = np.arange(0, duration, step=1 / fs)

    # - use a chirp signal
    f_min, f_max = freq_range
    period = time_test[-1]
    freq_inst = f_min + (f_max - f_min) * (time_test % period) / period
    phase_inst = 2 * np.pi * np.cumsum(freq_inst) * 1 / fs

    sig_test = np.sin(phase_inst)

    # - change to random signal
    # order = 2
    # cutoff = freq_range
    # b, a = butter(order, cutoff, output="ba", analog=False, btype="pass", fs=fs)
    # noise = np.random.randn(len(time_temp))
    # sig_temp = lfilter(b, a, noise)

    # - use a sinusoid signal
    freq_test = (f_min + f_max) / 2
    time_test = np.arange(0, duration_test, step=1 / fs)
    sig_test = np.sin(2 * np.pi * freq_test * time_test)

    # random DoA index
    rand_doa_index = int(np.random.rand(1)[0] * len(doa_list))
    rand_doa_index = len(doa_list) // 2  # len(doa_list) // 4
    doa_target = doa_list[rand_doa_index]

    # use fixed doa for the test
    doa_target = 0.0

    # compute the input signal received from the array
    sig_in = signal_from_template(template=(time_test, sig_test, doa_target), geometry=geometry)

    snr_gain_due_to_bandwidth = (fs / 2) / (f_max - f_min)
    snr_db_vec = [-10, 0, 10, 20]

    mm = 1 / 25.4

    plt.figure(figsize=[40 * mm, 40 * mm])

    filename = os.path.join(dir_name, "xylo_fixed_target_beam.pdf")

    for ind, snr_db in enumerate(snr_db_vec):
        # modify SNR to take bandwidth into account
        snr_db_bandwidth = snr_db - 10 * np.log10(snr_gain_due_to_bandwidth)

        snr_bandwidth = 10 ** (snr_db_bandwidth / 10)

        # add noise to the test signal
        sig_pow = np.mean(sig_in ** 2)
        noise_sigma = np.sqrt(sig_pow / snr_bandwidth)

        sig_in_noisy = sig_in + noise_sigma * np.random.randn(*sig_in.shape)

        # apply spike encoding to get the spikes
        spikes_in = xylosim.spike_encoding(sig_in=sig_in_noisy)

        # process the spikes by XyloSim SNN and produce output spikes
        spikes_out = xylosim.xylo_process(spikes_in=spikes_in)

        # compute the spike rate profile as a measure of power
        power = np.mean(spikes_out, axis=0) * fs
        power /= power.max()

        plt.plot(
            doa_list / np.pi * 180, 10 * np.log10(power), label=f"snr: {snr_db} dB",
            color=f'C{ind}',
        )
        plt.xlabel("DoA (deg.)")

        plt.axvline(x=doa_list[np.argmax(power)] / np.pi * 180, color=f'C{ind}', label="target DoA", linestyle='--', )

    plt.ylabel("Normalized Power (dB)")
    plt.xticks([-180, -90, 0, 90, 180])
    # plt.ylim([0, 1.05])
    # plt.legend()
    plt.grid(False)

    # plt.title(
    #     f"Angular power spectrum after beamforming:\nfreq-range:[{f_min / 1000:0.1f}, {f_max / 1000:0.1f}] KHz, DoA-target: {doa_target * 180 / np.pi:0.2f} deg"
    # )
    plt.axvline(x=0., color="k", label="target DoA", linestyle=':')
    # plt.legend()

    if SAVE_PLOTS:
        plt.savefig(filename, bbox_inches="tight", transparent=True)
    else:
        plt.draw()

    # apply statistical analysis of the precision of DoA estimation
    snr_db_vec = np.linspace(-10, 20, 11)
    # snr_db_vec = np.linspace(0,20, 5)
    angle_err = []

    test_duration = 1000e-3
    time_test = np.arange(0, test_duration, step=1 / fs)
    sig_test = np.sin(2 * np.pi * freq_design * time_test)

    # NOTE: use another signal for test since sinusoids at higher frequency can be problematic due to zero-crossing
    # - use a chirp signal
    f_min, f_max = freq_range
    period = time_test[-1]
    freq_inst = f_min + (f_max - f_min) * (time_test % period) / period
    phase_inst = 2 * np.pi * np.cumsum(freq_inst) * 1 / fs

    sig_test = np.sin(phase_inst)

    print("\n", " statistical analysis ".center(150, "*"), "\n")

    doa_err_vec = np.zeros((np.size(snr_db_vec), num_sim))

    for snr_ind, snr_db in tqdm(enumerate(snr_db_vec)):
        print(f"statistical analysis for snr-db: {snr_db}\n")

        # snr correction considering the snr improvement after filtering
        snr_db_target = snr_db - 10 * np.log10(snr_gain_due_to_bandwidth)
        snr_target = 10 ** (snr_db_target / 10)

        for sim in tqdm(range(num_sim)):
            doa_target = np.random.rand(1)[0] * 2 * np.pi

            # obtain the corresponding input signal in the input of array
            sig_in = signal_from_template(template=(time_test, sig_test, doa_target), geometry=geometry)

            # add noise to the signal
            sig_pow = np.mean(sig_in ** 2)
            noise_sigma = np.sqrt(sig_pow / snr_target)

            sig_in_noisy = sig_in + noise_sigma * np.random.randn(*sig_in.shape)

            # apply spike encoding to get the spikes
            spikes_in = xylosim.spike_encoding(sig_in=sig_in_noisy)

            # process the spikes by XyloSim SNN and produce output spikes
            spikes_out = xylosim.xylo_process(spikes_in=spikes_in)

            # compute the spike rate profile as a measure of power
            power = np.mean(spikes_out, axis=0) * fs
            power /= power.max()

            # estimate the DoA from power
            # method 1: without any denoising
            doa_target_est = doa_list[np.argmax(power)]

            # method 2: more robust for finding the location of peaks in power vector
            win_size = num_grid//32
            win_size = 2 * (win_size//2) + 1
            doa_target_index = find_peak_location(sig_in=power, win_size=win_size)
            doa_target_est = doa_list[doa_target_index]

            doa_err = np.arcsin(np.abs(np.sin(doa_target_est - doa_target)))
            doa_err_vec[snr_ind, sim] = doa_err

        # compute MAE error
        doa_err_avg = np.mean(np.abs(doa_err))

        angle_err.append(doa_err_avg)

    # plot
    # NOTE: we find the best monotone approximation
    angle_err = np.asarray(angle_err)
    # angle_err = approx_decreasing(np.asarray(angle_err))

    plt.figure(figsize=[40 * mm, 40 * mm])

    plt.plot(snr_db_vec, angle_err / np.pi * 180)
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Angle error (deg.)")

    num_xticks = 5

    filename = os.path.join(dir_name, "xylo_fixed_target_accuracy_snr.pdf")

    plt.figure(figsize=[40 * mm, 40 * mm])

    plt.boxplot(doa_err_vec.T / np.pi * 180, vert=True, labels=snr_db_vec, manage_ticks=False,
                medianprops={'linestyle': '-', 'color': 'black', 'linewidth': 0.5}, boxprops={'linewidth': 0.5},
                showcaps=False, sym='k.', flierprops={'markersize': 2}, whiskerprops={'linewidth': 0.5})
    plt.plot([0, len(snr_db_vec) + 1], [1, 1], 'k:', linewidth=0.5)
    # plt.xticks(range(1, len(doa_err_vec), 5))
    xticks = np.linspace(1, len(doa_err_vec), num_xticks)
    print(xticks)
    plt.xticks(
        xticks,
        [f'{t:0.1f}' for t in np.linspace(np.min(snr_db_vec), np.max(snr_db_vec), num_xticks)],
    )
    plt.xlabel("SNR (dB)")
    plt.ylabel("Angle error (deg.)")
    plt.ylim([-2, 20])

    print(f'SNR: {snr_db_vec}')
    print(f'Mean absolute errors: {np.mean(doa_err_vec, axis=1) * 180 / np.pi}')

    if not SAVE_PLOTS:
        plt.draw()
    else:
        plt.savefig(filename, bbox_inches="tight", transparent=True)

    if not SAVE_PLOTS:
        plt.show()


def test_moving_target():
    """
    this function tracks the performance of localization for a moving target.
    """
    # build a geometry
    radius = 4.5e-2
    num_mic = 7
    fs = 48_000

    freq_design = 2_000
    freq_range = [0.5 * freq_design, freq_design]

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 10.0e-3

    tau_mem = 1 / (2 * np.pi * freq_design)
    tau_syn = tau_mem
    tau_vec = np.asarray([tau_syn, tau_mem])

    # type of spike encoding
    bipolar_spikes = True

    # list of DoAs
    num_grid = 64 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    # duration of the signal used for
    duration = 1000e-3

    # use Demo as a dummy object for computation
    # NOTE: beamforming matrices are produced within this module
    xylosim = Demo(
        geometry=geometry,
        freq_bands=[freq_range],
        doa_list=doa_list,
        recording_duration=duration,
        kernel_duration=kernel_duration,
        bipolar_spikes=bipolar_spikes,
        xylosim_version=True,
        fs=fs,
    )

    # use the beamforming matrix for tracking a moving target
    snr_db = 100
    freq_ratio = 1.0
    freq_test = freq_ratio * freq_design

    duration_test = 5000e-3
    time_test = np.arange(0, duration_test, step=1 / fs)

    # a chirp signal
    f_min, f_max = freq_range
    period = duration_test / 5
    freq_inst = f_min + (f_max - f_min) * (time_test % period) / period
    phase_inst = 2 * np.pi * np.cumsum(freq_inst) * 1 / fs
    sig_test = np.sin(phase_inst)

    doa_max = np.pi * 0.5
    num_period = 0.5
    doa_test = doa_max * np.sin(num_period * np.pi / duration_test * time_test)

    # build the signal at the input of the array 
    sig_in = signal_from_template(template=(time_test, sig_test, doa_test), geometry=geometry)

    # add noise
    snr_gain_due_to_bandwidth = (fs / 2) / (f_max - f_min)
    snr_db_target = snr_db - 10 * np.log10(snr_gain_due_to_bandwidth)

    snr_target = 10 ** (snr_db_target / 10)
    sig_pow = np.mean(sig_in ** 2)

    noise_sigma = np.sqrt(sig_pow / snr_target)
    sig_in_noisy = sig_in + noise_sigma * np.random.randn(*sig_in.shape)

    # apply spike encoding to get the spikes
    spikes_in = xylosim.spike_encoding(sig_in=sig_in_noisy)

    # process the spikes by XyloSim SNN and produce output spikes
    spikes_out = xylosim.xylo_process(spikes_in=spikes_in)

    # we can consider spikes_out as the beamformed signal
    sig_bf = spikes_out

    # compute the envelope of output signal
    rise_time = 40e-3
    fall_time = 200e-3
    env = Envelope(rise_time=rise_time, fall_time=fall_time, fs=fs)

    # detect DoA based on signal envelop
    sig_bf_env = env.evolve(sig_bf)

    # compute the estimated DoA
    doa_index = np.argmax(sig_bf_env, axis=1)
    doa_est = doa_list[doa_index]

    plt.figure(figsize=(16, 10))
    plt.subplot(211)
    plt.plot(time_test, doa_est[: len(time_test)] * 180 / np.pi, label="estimated")
    plt.plot(time_test + kernel_duration, doa_test * 180 / np.pi, label="true")
    # plt.ylim([doa_list[0] * 180 / np.pi, doa_list[-1] * 180 / np.pi])
    plt.title(
        f"tracking a moving target: radius:{radius:0.3f}m, num-mic:{num_mic}\n"
        + f"freq-design:{int(freq_design)}, freq-test:{int(freq_test)}, ker-H duration:{1000 * kernel_duration:0.1f} ms, num-samples:{int(kernel_duration * fs)}"
    )
    plt.legend()
    plt.xlabel("time (sec)")
    plt.ylabel("DoA")

    doa_min = (np.min(doa_test) - 0.1) * 180 / np.pi
    doa_max = (np.max(doa_test) + 0.1) * 180 / np.pi
    plt.ylim([doa_min, doa_max])
    plt.grid(True)

    plt.show()


def main():
    test_speech_target()
    test_noisy_target()
    # test_moving_target()


if __name__ == "__main__":
    main()
