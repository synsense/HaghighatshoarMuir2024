# ----------------------------------------------------------------------------------------------------------------------
# This module includes several test scenarios for multi-mic localization using SNN encoding.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 31.08.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from numpy.linalg import norm
from scipy.signal import hilbert, medfilt, lfilter, butter
from micloc.array_geometry import CenterCircularArray
from micloc.snn_beamformer import SNNBeamformer, SNNBeamformerReal,SNNBeamformerSpiky
from micloc.utils import Envelope
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter


def test_array_resolution():
    """
    this function computes the array resolution curve at various angle via direct numerical simulation.
    """
    # build a geometry
    radius = 4.5e-2
    num_mic = 7
    fs = 48_000
    freq_design = 2_000

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 10e-3
    target_spike_rate = 2_000

    tau_mem = 1 / (2 * np.pi * freq_design)
    tau_syn = tau_mem
    tau_vec = np.asarray([tau_syn, tau_mem])

    beamf = SNNBeamformerSpiky(geometry=geometry, kernel_duration=kernel_duration, tau_vec=tau_vec,
                          target_spike_rate=target_spike_rate, fs=fs)

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 1.0
    time_temp = np.arange(0, duration, step=1 / fs)

    # additive noise for the signal
    snr_db_design = 1_500
    snr = 10 ** (snr_db_design / 10)

    signal_dict = dict()
    order = 2
    freq_min = 0.5 * freq_design
    cutoff = [freq_min, freq_design]
    b, a = butter(order, cutoff, btype='bandpass', analog=False, output='ba', fs=fs)
    noise = np.random.randn(len(time_temp))
    sig_temp = lfilter(b, a, noise)
    measure_noise = 1 / np.sqrt(np.mean(sig_temp ** 2)) * np.random.randn(len(time_temp))
    sig_temp += measure_noise

    signal_dict["filtered_noise"] = sig_temp

    # ordinary sinusoid
    sig_temp = np.sin(2 * np.pi * freq_design * time_temp)
    measure_noise = 1 / np.sqrt(np.mean(sig_temp ** 2)) * np.random.randn(len(time_temp))
    sig_temp += measure_noise

    signal_dict["sin"] = sig_temp

    # chirp
    freq_inst = freq_min + (freq_design - freq_min) * time_temp / time_temp[-1]
    phase_inst = 2 * np.pi * np.cumsum(freq_inst) * 1 / fs
    sig_temp = np.sin(phase_inst)
    measure_noise = 1 / np.sqrt(np.mean(sig_temp ** 2)) * np.random.randn(len(time_temp))
    sig_temp += measure_noise

    signal_dict["chirp"] = sig_temp

    # 2. use an angular grid
    num_grid = 32 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    sig_temp = signal_dict["filtered_noise"]
    bf_mat = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list)

    # apply the beamforming matrices to signal samples
    rand_index = int(np.random.rand(1)[0] * len(doa_list))
    doa_target = doa_list[rand_index]
    snr_db_evaluation = 20

    sig_temp = signal_dict["sin"]
    sig_bf = beamf.apply_to_template(bf_mat=bf_mat, template=(time_temp, sig_temp, doa_target * np.ones(len(sig_temp))),
                                     snr_db=snr_db_evaluation)

    sig_power = np.mean(np.abs(sig_bf) ** 2, axis=1)
    sig_power /= np.max(sig_power)

    # plot the array resolution
    corr = np.abs(bf_mat.conj().T @ bf_mat)

    selected_indices = np.arange(0, len(corr), len(corr) // 4)
    plt.figure()
    plt.plot(doa_list / np.pi * 180, corr[selected_indices, :].T)
    plt.xlabel("DoA")
    plt.ylabel("array resolution")
    plt.title(f"array resolution: freq={freq_design}, ker-duration:{kernel_duration} sec")
    plt.grid(True)

    plt.figure()
    plt.plot(doa_list / np.pi * 180, sig_power, label="power")
    plt.xlabel("DoA [degree]")
    plt.ylabel("power after beamforming")
    plt.axvline(x=doa_target * 180 / np.pi, label=f"target DoA={doa_target * 180 / np.pi:0.1f}", color="r")
    plt.legend()
    plt.title(f"power of the beamformed signal: snr:{snr_db_evaluation} dB")
    plt.grid(True)

    plt.show()


def test_fixed_target():
    """
    this function evaluates the localization performance for a fixed  and non-moving target.
    """
    # build a geometry
    radius = 4.5e-2
    num_mic = 7
    fs = 48_000

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 10.0e-3

    target_spike_rate = 2_000
    tau_mem = 1 / (2 * np.pi * target_spike_rate)
    tau_syn = tau_mem
    tau_vec = np.asarray([tau_syn, tau_mem])

    beamf = SNNBeamformerSpiky(geometry=geometry, kernel_duration=kernel_duration, tau_vec=tau_vec,
                          target_spike_rate=target_spike_rate, fs=fs)

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 1000e-3
    freq_design = 2_000
    time_temp = np.arange(0, duration, step=1 / fs)
    sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

    # 2. use an angular grid
    num_grid = 16 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    bf_mat = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list)

    # use the beamforming matrix
    freq_ratio = 1.00
    freq_test = freq_ratio * freq_design
    sig_temp = np.sin(2 * np.pi * freq_test * time_temp)

    doa_target = np.pi / 4
    snr_db = 10
    sig_bf = beamf.apply_to_template(bf_mat=bf_mat, template=(time_temp, sig_temp, doa_target), snr_db=snr_db)

    # compute power
    power = np.mean(np.abs(sig_bf) ** 2, axis=1)
    power /= power.max()

    plt.plot(doa_list / np.pi * 180, power, label="power density")
    plt.xlabel("DoA")
    plt.ylabel("power")
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.title(
        f"power of beamformed signal:\nfreq-design:{freq_design} Hz\nfreq-target:{freq_test} Hz, DoA-target: {doa_target * 180 / np.pi:0.2f}")
    plt.axvline(x=doa_target * 180 / np.pi, color="r", label="target DoA")

    plt.show()


def test_moving_target():
    """
    this function tracks the performance of localization for a moving target.
    """
    # build a geometry
    radius = 4.5e-2
    num_mic = 7
    fs = 48_000

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 10.0e-3

    target_spike_rate = 2_000
    tau_mem = 1 / (2 * np.pi * target_spike_rate)
    tau_syn = tau_mem
    tau_vec = np.asarray([tau_syn, tau_mem])

    beamf = SNNBeamformerSpiky(geometry=geometry, kernel_duration=kernel_duration, tau_vec=tau_vec,
                          target_spike_rate=target_spike_rate, fs=fs)

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duration = 1000e-3
    freq_design = 1500
    time_temp = np.arange(0, duration, step=1 / fs)
    sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

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
    sig_test = np.sin(2 * np.pi * freq_test * time_test)

    doa_max = np.pi * 0.9
    num_period = 0.4
    doa_test = doa_max * np.sin(num_period * np.pi / duration_test * time_test)

    sig_bf = beamf.apply_to_template(bf_mat=bf_mat, template=(time_test, sig_test, doa_test), snr_db=snr_db)

    # compute the envelope of output signal
    rise_time = 10e-3
    fall_time = 300e-3
    env = Envelope(rise_time=rise_time, fall_time=fall_time, fs=fs)

    sig_bf_env = env.evolve(sig_bf.T).T

    # compute the estimated DoA
    doa_index = np.argmax(sig_bf_env, axis=0)
    doa_est = doa_list[doa_index]

    fall_freq = 1/(2*np.pi*fall_time)
    order = 2
    cut_off = fall_freq
    b, a = butter(order, cut_off, btype="low", output="ba", analog=False, fs=fs)
    doa_est = lfilter(b, a, doa_est)

    rel_err = np.sqrt(
        np.median((doa_est - doa_test[:len(doa_est)]) ** 2) / (np.sqrt(np.median(doa_est ** 2) * np.median(doa_test ** 2))))
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
    # test_array_resolution()
    # test_fixed_target()
    test_moving_target()


if __name__ == '__main__':
    main()
