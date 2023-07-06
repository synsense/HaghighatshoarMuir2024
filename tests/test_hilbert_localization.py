# ----------------------------------------------------------------------------------------------------------------------
# This module includes several test scenarios for multi-mic localization.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.07.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np

from micloc.array_geometry import CircularArray
from micloc.beamformer import Beamformer
from micloc.utils import Envelope
import matplotlib.pyplot as plt
from scipy.signal import lfilter


def test_array_resolution():
    """
    this function computes the array resolution curve at various angle via direct numerical simulation.
    """
    # build a geometry
    radius = 5e-2
    num_mic = 16
    fs = 50_000

    geometry = CircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 100e-3
    beamf = Beamformer(geometry=geometry, kernel_duration=kernel_duration, fs=fs)

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duartion = 0.5
    freq_design = 4_000
    time_temp = np.arange(0, duartion, step=1 / fs)
    sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

    # 2. use an angular grid
    num_grid = 8 * num_mic
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    bf_mat = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list)

    # plot the array resolution
    corr = np.abs(bf_mat.conj().T @ bf_mat)

    selected_indices = [0, len(corr) // 2, len(corr) - 1]
    plt.plot(doa_list / np.pi * 180, corr[selected_indices, :].T)
    plt.xlabel("DoA")
    plt.ylabel("array resolution")
    plt.title(f"array resolution: freq={freq_design}, ker-duration:{kernel_duration} sec")
    plt.grid(True)
    plt.show()


def test_fixed_target():
    """
    this function evaluates the localization performance for a fixed  and non-moving target.
    """
    # build a geometry
    radius = 5e-2
    num_mic = 8
    fs = 50_000

    geometry = CircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 30e-3
    beamf = Beamformer(geometry=geometry, kernel_duration=kernel_duration, fs=fs)

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duartion = 30e-3
    freq_design = 4_000
    time_temp = np.arange(0, duartion, step=1 / fs)
    sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

    # 2. use an angular grid
    num_grid = 16 * num_mic
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    bf_mat = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list)

    # use the beamforming matrix
    freq_test = freq_design * 1.1

    doa_target = np.pi / 4
    sig_bf = beamf.apply_to_template(bf_mat=bf_mat, template=(time_temp, sig_temp, doa_target))

    # compute power
    power = np.mean(np.abs(sig_bf) ** 2, axis=1)
    power /= power.max()

    plt.plot(doa_list / np.pi * 180, power, label="power density")
    plt.xlabel("DoA")
    plt.ylabel("power")
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.title(
        f"power of beamformed signal: freq-design:{freq_design} Hz\nfreq-target:{freq_test} Hz, DoA-target: {doa_target * 180 / np.pi:0.2f}")
    plt.axvline(x=doa_target * 180 / np.pi, color="r", label="target DoA")

    plt.show()


def test_moving_target():
    """
    this function tracks the performance of localization for a moving target.
    """
    # build a geometry
    radius = 5e-2
    num_mic = 16
    fs = 50_000

    # signal to noise ratio in each array element
    snr_db = 10

    geometry = CircularArray(radius=radius, num_mic=num_mic)

    # build the corresponding beamformer
    kernel_duration = 3e-3
    beamf = Beamformer(geometry=geometry, kernel_duration=kernel_duration, fs=fs)

    # build beamformer matrix for various DoAs
    # 1. build a template signal
    duartion = 100e-3
    freq_design = 3_000
    time_temp = np.arange(0, duartion, step=1 / fs)
    sig_temp = np.sin(2 * np.pi * freq_design * time_temp)

    # 2. use an angular grid
    num_grid = 32 * num_mic
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    doa_resolution = (doa_list[-1] - doa_list[0]) / num_grid

    bf_mat = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list)

    # use the beamforming matrix for tracking a moving target
    freq_test = freq_design * 1.1
    duration_test = 3
    time_test = np.arange(0, duration_test, step=1 / fs)
    sig_test = np.sin(2 * np.pi * freq_test * time_test)

    doa_max = np.pi * 0.9
    doa_test = doa_max * np.sin(2 * np.pi / duration_test * time_test)

    sig_bf = beamf.apply_to_template(bf_mat=bf_mat, template=(time_test, sig_test, doa_test), snr_db=snr_db)

    # compute the envelope of output signal
    rise_time = 10e-3
    fall_time = 0.1
    env = Envelope(rise_time=rise_time, fall_time=fall_time, fs=fs)

    sig_bf_env = env.evolve(sig_bf.T).T

    # compute the estimated DoA
    doa_index = np.argmax(sig_bf_env, axis=0)
    doa_est = doa_list[doa_index]

    rel_err = np.sqrt(
        np.median((doa_est - doa_test[:-1]) ** 2) / np.sqrt(np.median(doa_est ** 2) * np.median(doa_test ** 2)))
    angle_err = (doa_est - doa_test[:-1]) * 180 / np.pi

    med_err = np.median(np.abs(angle_err))

    plt.figure(figsize=(16, 10))

    plt.subplot(211)
    plt.plot(time_test[:-1], doa_est * 180 / np.pi, label="estimated")
    plt.plot(time_test, doa_test * 180 / np.pi, label="true")
    plt.ylim([doa_list[0] * 180 / np.pi, doa_list[-1] * 180 / np.pi])
    plt.title(
        f"tracking a moving target: radius:{radius:0.3f}m, num-mic:{num_mic}\n" + \
        f"DoA resolution: {doa_resolution * 180 / np.pi:0.1f} deg, rel-err:{rel_err:0.5f}, med-err:{med_err:0.2f} deg\n" + \
        f"freq-design:{int(freq_design)}, freq-test:{int(freq_test)}, ker-H duration:{1000 * kernel_duration:0.1f} ms, num-samples:{int(kernel_duration * fs)}"
    )
    plt.legend()
    plt.xlabel("time (sec)")
    plt.ylabel("DoA")
    plt.grid(True)

    plt.subplot(212)
    plt.plot(np.sort(angle_err), 1 - np.linspace(0, 1, len(angle_err)))
    plt.grid(True)
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
