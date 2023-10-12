# ----------------------------------------------------------------------------------------------------------------------
# This module uses SNN beamforming to localize and track real targets.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 26.09.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import cvxpy as cp
from numpy.linalg import norm
from scipy.signal import hilbert, medfilt, lfilter, butter
from micloc.array_geometry import CenterCircularArray
from micloc.snn_beamformer import SNNBeamformer
from micloc.utils import Envelope
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import lfilter, butter
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


SAVE_PLOTS = False

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


def test_fixed_target():
    """
    this function evaluates the localization performance for a fixed  and non-moving target.
    """

    dir_name = os.path.join(Path(__file__).resolve().parent, "fixed_target")
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

    tau_mem = 1 / (2 * np.pi * freq_design)
    tau_syn = tau_mem
    tau_vec = np.asarray([tau_syn, tau_mem])

    beamf = SNNBeamformer(geometry=geometry, kernel_duration=kernel_duration, tau_vec=tau_vec, freq_range=freq_range,
                          fs=fs)

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 1000e-3
    time_temp = np.arange(0, duration, step=1 / fs)
    # sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

    # build a chirp signal
    f_min, f_max = freq_range
    period = time_temp[-1]
    freq_inst = f_min + (f_max - f_min) * (time_temp % period) / period
    phase_inst = 2 * np.pi * np.cumsum(freq_inst) * 1 / fs

    sig_temp = np.sin(phase_inst)

    # 2. use an angular grid
    num_grid = 64 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    bf_mat = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list)

    freq_test = (f_min + f_max) / 2
    time_test = time_temp[:len(time_temp) // 16]
    sig_test = np.sin(2 * np.pi * freq_test * time_test)

    # random DoA index
    rand_doa_index = int(np.random.rand(1)[0] * len(doa_list))
    rand_doa_index = len(doa_list) // 2
    doa_target = doa_list[rand_doa_index]

    snr_gain_due_to_bandwidth = (fs / 2) / (f_max - f_min)
    snr_db_vec = [-10, 0, 10, 20]
    plt.figure()

    filename = os.path.join(dir_name, "fixed_target_beam.pgf")

    for snr_db in snr_db_vec:
        # modify SNR to take bandwidth into account
        snr_db_bandwidth = snr_db - 10 * np.log10(snr_gain_due_to_bandwidth)
        sig_bf = beamf.apply_to_template(bf_mat=bf_mat, template=(time_test, sig_test, doa_target),
                                         snr_db=snr_db_bandwidth)

        # compute power
        power = np.mean(np.abs(sig_bf) ** 2, axis=1)
        power /= power.max()

        plt.plot(doa_list / np.pi * 180, 10 * np.log10(power), label=f"snr: {snr_db} dB")
        plt.xlabel("DoA")

    plt.ylabel("Normalized Power [dB]")
    # plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(True)
    plt.title(
        f"Angular power spectrum after beamforming:\nfreq-range:[{f_min / 1000:0.1f}, {f_max / 1000:0.1f}] KHz, DoA-target: {doa_target * 180 / np.pi:0.2f} deg")
    plt.axvline(x=doa_target * 180 / np.pi, color="r", label="target DoA")

    if SAVE_PLOTS:
        plt.savefig(filename)
    else:
        plt.draw()

    # apply statistical analysis of the precision of DoA estimation
    snr_db_vec = np.linspace(-10, 10, 20)
    angle_err = []
    num_sim = 300

    test_duration = 100e-3
    time_test = np.arange(0, test_duration, step=1 / fs)
    sig_test = np.sin(2 * np.pi * freq_design * time_test)

    print("\n", " statistical analysis ".center(150, "*"), "\n")

    for snr_db in tqdm(snr_db_vec):
        doa_err_vec = []

        # snr correction considering the snr improvement after filtering
        snr_db_target = snr_db - 10 * np.log10(snr_gain_due_to_bandwidth)

        for sim in range(num_sim):
            doa_target = np.random.rand(1)[0] * 2 * np.pi

            # extract the beamformed signal
            sig_bf = beamf.apply_to_template(bf_mat=bf_mat, template=(time_test, sig_test, doa_target),
                                             snr_db=snr_db_target)

            # power after beamforming
            power = np.mean(np.abs(sig_bf) ** 2, axis=1)

            doa_target_est = doa_list[np.argmax(power)]

            doa_err = np.arcsin(np.abs(np.sin(doa_target_est - doa_target)))
            doa_err_vec.append(doa_err)

        # compute MMSE error
        doa_err_avg = np.sqrt(np.median(doa_err ** 2))

        angle_err.append(doa_err_avg)

    # plot
    # NOTE: we fins the best monotone approximation
    angle_arr = np.asarray(angle_err)
    angle_err = approx_decreasing(np.asarray(angle_err))

    plt.figure()
    plt.plot(snr_db_vec, angle_err / np.pi * 180)
    plt.grid(True)
    plt.xlabel("SNR [dB]")
    plt.ylabel("angle error [deg]")

    if not SAVE_PLOTS:
        plt.draw()
    else:
        plt.savefig(filename)

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

    beamf = SNNBeamformer(geometry=geometry, kernel_duration=kernel_duration, tau_vec=tau_vec, freq_range=freq_range,
                          fs=fs)

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 1000e-3
    time_temp = np.arange(0, duration, step=1 / fs)
    sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

    # build a chirp signal
    f_min, f_max = freq_range
    period = time_temp[-1]
    freq_inst = f_min + (f_max - f_min) * (time_temp % pariod) / period
    phase_inst = np.cumsum(freq_inst) * 1 / fs

    sig_temp = np.sin(phase_inst)

    # 2. use an angular grid
    num_grid = 32 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    bf_mat = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list)

    # use the beamforming matrix for tracking a moving target
    snr_db = 100
    freq_ratio = 1.0
    freq_test = freq_ratio * freq_design

    duration_test = 5000e-3
    time_test = np.arange(0, duration_test, step=1 / fs)
    sig_test = sig_temp

    doa_max = np.pi * 0.9
    num_period = 0.4
    doa_test = doa_max * np.sin(num_period * np.pi / duration_test * time_test)

    sig_bf = beamf.apply_to_template(bf_mat=bf_mat, template=(time_test, sig_test, doa_test), snr_db=snr_db)

    # compute the envelope of output signal
    rise_time = 10e-3
    fall_time = 100e-3
    env = Envelope(rise_time=rise_time, fall_time=fall_time, fs=fs)

    sig_bf_env = env.evolve(sig_bf.T).T

    # compute the estimated DoA
    doa_index = np.argmax(sig_bf_env, axis=0)
    doa_est = doa_list[doa_index]

    # fall_freq = 1/(2*np.pi*fall_time)
    # order = 2
    # cut_off = fall_freq
    # b, a = butter(order, cut_off, btype="low", output="ba", analog=False, fs=fs)
    # doa_est = lfilter(b, a, doa_est)

    rel_err = np.sqrt(
        np.median((doa_est - doa_test[:len(doa_est)]) ** 2) / (
            np.sqrt(np.median(doa_est ** 2) * np.median(doa_test ** 2))))
    angle_err = (doa_est - doa_test[:len(doa_est)]) * 180 / np.pi

    med_err = np.median(np.abs(angle_err))

    plt.figure(figsize=(16, 10))

    plt.subplot(211)
    plt.plot(time_test, doa_est[:len(time_test)] * 180 / np.pi, label="estimated")
    plt.plot(time_test + kernel_duration, doa_test * 180 / np.pi, label="true")
    plt.ylim([doa_list[0] * 180 / np.pi, doa_list[-1] * 180 / np.pi])
    plt.title(
        f"tracking a moving target: radius:{radius:0.3f}m, num-mic:{num_mic}\n" + \
        f"rel-err:{rel_err:0.5f}, med-err:{med_err:0.2f} deg\n" + \
        f"freq-design:{int(freq_design)}, freq-test:{int(freq_test)}, ker-H duration:{1000 * kernel_duration:0.1f} ms, num-samples:{int(kernel_duration * fs)}"
    )
    plt.legend()
    plt.xlabel("time (sec)")
    plt.ylabel("DoA")
    plt.grid(True)

    plt.subplot(212)
    plt.plot(np.sort(angle_err), 1 - np.linspace(0, 1, len(angle_err)))
    plt.grid(True)
    doa_resolution = np.diff(doa_list)[0]
    doa_range = 8 * doa_resolution * 180 / np.pi
    plt.xlim([-doa_range, doa_range])
    plt.xlabel(f"angle deg [median of error {med_err:0.2f} deg]")
    plt.ylabel("CDF of error")

    plt.show()


def main():
    test_fixed_target()
    # test_moving_target()


if __name__ == '__main__':
    main()