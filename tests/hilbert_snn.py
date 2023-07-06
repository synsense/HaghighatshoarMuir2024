# ---------------------------------------------------------------------------
# This module investigates the implementation of the Hilbert transform in SNN
# domain.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 10.03.2023
# ---------------------------------------------------------------------------

import numpy as np
from scipy.signal import hilbert, lfilter, freqz, butter
import matplotlib.pyplot as plt

from tqdm import tqdm


def test_hilbert_harmonics():
    """
    this function investigates the Hilbert transform for harmonic signals.
    """
    freq = 100
    num_periods = 2
    duration = num_periods / freq

    oversampling = 20
    fs = oversampling * freq

    time_vec = np.arange(0, duration, step=1 / fs)
    sig_in = np.sin(2 * np.pi * freq * time_vec)

    sig_in_h = hilbert(sig_in)

    plt.plot(figsize=(16, 10))
    plt.plot(np.real(sig_in_h))
    plt.plot(np.imag(sig_in_h))

    plt.grid(True)
    plt.show()


def test_hilbert_SNN():
    alpha = 0.9

    size_vec = [100, 200, 300, 400, 500]

    plt.figure(figsize=(16, 10))

    for size in size_vec:
        # impulse response of Synapse + Neuron
        sig_in = np.arange(size) * alpha ** np.arange(size)

        # hilbert transform
        sig_in_h = hilbert(sig_in)

        time_vec = np.arange(size)

        plt.plot(time_vec, np.real(sig_in_h))
        plt.plot(time_vec, np.imag(sig_in_h))

    plt.xlabel("time (sec)")
    plt.ylabel("signal and hilbert transform")
    plt.grid(True)
    plt.show()


def test_hilbert_SNN_localization():
    num_mic = 8
    freq = 1000
    speed = 340
    wavelength = speed / freq
    spacing = wavelength / 2 * np.arange(num_mic)

    angle = -np.pi / 4
    delay_vec = spacing * np.sin(angle) / speed
    delay_vec = delay_vec - np.min(delay_vec)

    delay_jump = 4 * np.mean(delay_vec)
    delay_jump_vec = delay_jump * np.arange(10)

    fs = 10 * freq
    alpha = freq
    duration = 1.2 * np.max(delay_jump_vec)
    time_vec = np.arange(0, duration, step=1 / fs)

    # signal received from various array elements
    sig_in_vec = []
    for delay in delay_vec:
        sig_in = 0
        for jump in delay_jump_vec:
            time_delay = time_vec - jump - delay
            sig_in = sig_in + time_delay * np.exp(-alpha * time_delay) * np.heaviside(time_delay, 0)
            # sig_in = sig_in + np.sin(2*np.pi*freq*time_delay)

        sig_in_vec.append(sig_in)

    sig_in_vec = np.asarray(sig_in_vec)

    # compute the Hilbert transform
    sig_in_vec_h = hilbert(sig_in_vec, axis=1)

    # compute the SVD to localize
    cov_mat = sig_in_vec_h @ np.conjugate(sig_in_vec_h).T / sig_in_vec_h.shape[1]
    U, D, V = np.linalg.svd(cov_mat)

    # plot the array response vec
    psd_angle = np.fft.fftshift(np.abs(np.fft.fft(U[:, 0], 20 * len(U))))
    angle_vec = -np.arcsin(np.linspace(-1, 1, len(psd_angle))) * 180 / np.pi

    plt.subplot(511)
    plt.plot(D)
    plt.xlabel("svd index")
    plt.ylabel("svd values")
    plt.title("subspace method for array processing")

    plt.subplot(512)
    plt.plot(np.real(U[:, 0]))
    plt.plot(np.imag(U[:, 0]))

    plt.subplot(513)
    plt.plot(angle_vec, psd_angle)
    plt.axvline(x=angle * 180 / np.pi, color='r', linewidth=3)

    plt.subplot(514)
    plt.plot(time_vec, np.real(sig_in_vec_h).T)

    plt.subplot(515)
    plt.plot(time_vec, np.imag(sig_in_vec_h).T)

    plt.show()


def test_hilbert_SNN_beamforming():
    """
    this module builds beamforming vectors for an array using hilbert transform method.
    """

    # array specs
    num_mic = 32
    freq = 1000
    speed = 340
    wavelength = speed / freq
    spacing = wavelength / 2 * np.arange(num_mic)

    # SNN specs
    fs = 50_000
    alpha = 4 * freq

    # number of angles in DoA grid
    num_angle = 16 * num_mic + 1
    angle_vec = np.linspace(-np.pi / 2, np.pi / 2, num_angle)

    # beamforming vectors
    bf_mat = []

    for angle in tqdm(angle_vec):
        # print("building beamforming vector for angle ", angle)

        # build the delay vector corresponding to this angle
        delay_vec = spacing * np.sin(angle) / speed
        delay_vec = delay_vec - np.min(delay_vec)

        # delay_jump = 2*np.max(delay_vec)
        delay_jump = 2 / freq
        delay_jump_vec = delay_jump * np.arange(40)

        duration = 1.2 * np.max(delay_jump_vec)
        time_vec = np.arange(0, duration, step=1 / fs)

        # signal received from various array elements
        sig_in_vec = []
        for delay in delay_vec:
            sig_in = 0
            for jump in delay_jump_vec:
                time_delay = time_vec - jump - delay
                sig_in = sig_in + time_delay * np.exp(-alpha * time_delay) * np.heaviside(time_delay, 0)
                # sig_in = sig_in + np.sin(2*np.pi*freq*time_delay)

            sig_in_vec.append(sig_in)

        sig_in_vec = np.asarray(sig_in_vec)

        # compute the Hilbert transform
        sig_in_vec_h = hilbert(sig_in_vec, axis=1)

        # remove the transient part of hilbert transform
        T = sig_in_vec_h.shape[1]
        sig_in_vec_h_stable = sig_in_vec_h[:, T // 4:-T // 4]

        # compute the SVD to localize
        cov_mat = sig_in_vec_h_stable @ np.conjugate(sig_in_vec_h_stable).T / sig_in_vec_h_stable.shape[1]
        U, D, V = np.linalg.svd(cov_mat)

        bf_vec = U[:, 0]

        bf_mat.append(bf_vec)

    bf_mat = np.asarray(bf_mat).T

    # plot the last set of waveforms
    plt.figure(figsize=(16, 10))
    plt.plot(time_vec, np.real(sig_in_vec_h).T, label='in-phase')
    plt.plot(time_vec, np.imag(sig_in_vec_h).T, label='quadrature')
    plt.xlabel("time (sec)")
    plt.ylabel("waveforms")
    plt.grid(True)
    plt.show(block=False)

    # * build the correlation matrix
    # the real version can be implemented easily in SNN
    # corr = np.abs(np.real(np.conjugate(bf_mat).T @ bf_mat))

    # the imaginary version is better suited since it combines real and imaginary parts
    corr = np.abs(np.conjugate(bf_mat).T @ bf_mat)

    corr_center = corr[[0, len(corr) // 2, len(corr) - 1]].T

    plt.figure(figsize=(16, 10))
    plt.plot(angle_vec * 180 / np.pi, corr_center)
    plt.legend(['left', 'center', 'right'])
    plt.xlabel("angle [deg]")
    plt.ylabel("spatial correlation")
    plt.grid(True)
    plt.title(f"spatial resolution curve at various angular sections (num mic: {num_mic})")
    plt.ylim([0, 1.05])

    plt.show()


def test_hilbert_kernel():
    """
    this module evaluates how Hilbert kernels are in the time domain.
    """
    # signal length
    L = 101
    x = np.zeros(L)
    x[0] = 1

    xh = hilbert(x)

    xh_imag_est = np.zeros(L)
    xh_imag_est[1:] = 2 * np.sin(np.pi * np.arange(1, L) / 2) ** 2 / (np.pi * np.arange(1, L))

    plt.figure(figsize=(16, 10))
    plt.plot(np.real(xh))
    plt.plot(np.imag(xh))
    plt.plot(xh_imag_est, '+')
    plt.xlabel("index")
    plt.ylabel("Hilbert kernels")
    plt.title(f"Hilbert transform kernels of size {L}")
    plt.grid(True)

    plt.show()


def test_short_hilbert():
    """
    this function tests how hilbert transform can be used for short sections of signal.
    """

    # * some random signal
    freq = 100
    num_period = 8
    duration = num_period / freq

    fs = 16 * freq
    time_vec = np.arange(0, duration, step=1 / fs)
    sig_in = np.sin(2 * np.pi * freq * time_vec)

    # * build the kernel
    num_section_period = 5
    kernel_duration = num_section_period / freq
    L = int(kernel_duration * fs)
    impulse = np.zeros(L)
    impulse[0] = 1

    xh = hilbert(impulse)

    # real and imag kernels
    ker_real = np.real(xh)
    ker_imag = np.imag(xh)

    # * compute the Hilbert transform in two ways
    sig_in_h_short = np.convolve(sig_in, ker_real) + 1j * np.convolve(sig_in, ker_imag)
    sig_in_h = hilbert(sig_in)

    # * compute the recovered phases
    phase_h = np.unwrap(np.angle(sig_in_h))
    phase_h_short = np.unwrap(np.angle(sig_in_h_short[:len(sig_in_h)]))

    plt.figure(figsize=(16, 10))

    plt.subplot(211)
    plt.plot(time_vec, np.imag(sig_in_h), color='b', label='full')
    plt.plot(time_vec, np.real(sig_in_h), color='black', label='original')
    plt.plot(time_vec, np.imag(sig_in_h_short[:len(time_vec)]), color='r', label='short')
    plt.title(f"kernel duration: {kernel_duration:0.4f} sec, length: {L} samples")
    plt.legend()
    plt.grid(True)
    plt.ylabel("hilbert transforms")

    plt.subplot(212)
    plt.plot(time_vec, phase_h, label="original")
    plt.plot(time_vec, phase_h_short, label="short")
    plt.grid(True)
    plt.legend()
    plt.xlabel("time (sec)")
    plt.ylabel("phases")

    plt.show()


def test_hilbert_chirp():
    """
    this function investigates the localization when the input signal is a chirp.
    """
    # array information
    speed = 340
    num_mic = 16
    freq_min = 1_000
    freq_max = 1001

    lambda_min = speed / freq_max
    spacing = lambda_min / 2
    geometry = spacing * np.arange(num_mic)

    # signal information
    period = 0.5
    duration = 2 * period
    fs = 50_000
    time_vec = np.arange(0, duration, step=1 / fs)

    # angular grid
    num_grid = 8 * num_mic + 1
    angle_vec = np.linspace(-np.pi / 2, np.pi / 2, num_grid)

    # build the template for chirp freq and phase pattern
    # instantenous freq
    freq_inst = freq_min + (freq_max - freq_min) * np.mod(time_vec, period) / period
    phase_inst = 2 * np.pi * np.cumsum(freq_inst) / fs

    bf_mat = []

    for angle in tqdm(angle_vec):
        # produce the corresponding delays
        delay_vec = geometry * np.sin(angle) / speed

        # produce the signal
        sig_in_vec = []

        for delay in delay_vec:
            # delayed time
            time_delay = time_vec - delay
            time_delay[time_delay < 0] = 0

            # apply interpolation to recover the delayed phase
            phase = np.interp(time_delay, time_vec, phase_inst)
            sig_in = np.sin(phase)

            sig_in_vec.append(sig_in)

        sig_in_vec = np.asarray(sig_in_vec)

        # compute the hilbert transform
        sig_in_vec_h = hilbert(sig_in_vec, axis=1)

        # compute the SVD for beamforming
        cov_mat = sig_in_vec_h @ np.conjugate(sig_in_vec_h).T / sig_in_vec_h.shape[0]
        U, D, _ = np.linalg.svd(cov_mat)

        # register the corresponding beamforming vector
        bf_mat.append(U[:, 0])

    # compute the array resolution
    bf_mat = np.conjugate(np.asarray(bf_mat))

    corr = bf_mat @ np.conjugate(bf_mat).T

    step = len(corr) // 4
    corr_selected = np.abs(corr[::step, :])

    plt.figure(figsize=(16, 10))
    plt.plot(angle_vec / np.pi * 180, corr_selected.T)
    plt.grid(True)
    plt.xlabel("angle [deg]")
    plt.ylabel("array resolution")
    plt.title(f"Hilbert: array resolution for a chirp in freq range [{freq_min:0.0f}, {freq_max:0.0f}] Hz")
    plt.ylim([0, 1.05])

    plt.show(block=False)

    # ===========================================================================
    # *       Repeat the simulation by using a small Hilbert kernel
    # ===========================================================================

    # build the kernel
    ker_duration = period / 100
    ker_len = int(ker_duration * fs)

    impulse = np.zeros(ker_len)
    impulse[0] = 1

    ker_hilbert = np.imag(hilbert(impulse))

    # build the beamforming matrices using the kernel
    bf_mat = []

    for angle in tqdm(angle_vec):
        # produce the corresponding delays
        delay_vec = geometry * np.sin(angle) / speed

        # produce the signal
        sig_in_vec = []

        for delay in delay_vec:
            # delayed time
            time_delay = time_vec - delay
            time_delay[time_delay < 0] = 0

            # apply interpolation to recover the delayed phase
            phase = np.interp(time_delay, time_vec, phase_inst)
            sig_in = np.sin(phase)

            sig_in_vec.append(sig_in)

        sig_in_vec = np.asarray(sig_in_vec)

        # compute the hilbert transform
        sig_in_vec_kernel = lfilter(ker_hilbert, [1], sig_in_vec, axis=1)[:, :sig_in_vec.shape[1]]
        sig_in_vec_h = sig_in_vec + 1j * sig_in_vec_kernel

        # compute the SVD for beamforming
        cov_mat = sig_in_vec_h @ np.conjugate(sig_in_vec_h).T / sig_in_vec_h.shape[0]
        U, D, _ = np.linalg.svd(cov_mat)

        # register the corresponding beamforming vector
        bf_mat.append(U[:, 0])

    # compute the array resolution
    bf_mat = np.conjugate(np.asarray(bf_mat))

    corr = bf_mat @ np.conjugate(bf_mat).T

    step = len(corr) // 4
    corr_selected = np.abs(corr[::step, :])

    plt.figure(figsize=(16, 10))
    plt.plot(angle_vec / np.pi * 180, corr_selected.T)
    plt.grid(True)
    plt.xlabel("angle [deg]")
    plt.ylabel("array resolution")
    plt.title(
        f"Kernel Hilbert: array resolution for a chirp in freq range [{freq_min:0.0f}, {freq_max:0.0f}] Hz, ker-duartion: {ker_duration}, ker-len:{ker_len} samples")
    plt.ylim([0, 1.05])

    plt.show(block=False)

    # investigate the effect of beamforming in terms of amplitude
    angle = np.pi / 8

    # produce the corresponding delays
    delay_vec = geometry * np.sin(angle) / speed

    # produce the signal
    sig_in_vec = []

    for delay in delay_vec:
        # delayed time
        time_delay = time_vec - delay
        time_delay[time_delay < 0] = 0

        # apply interpolation to recover the delayed phase
        phase = np.interp(time_delay, time_vec, phase_inst)
        sig_in = np.sin(phase)

        sig_in_vec.append(sig_in)

    sig_in_vec = np.asarray(sig_in_vec)

    # compute the hilbert transform
    sig_in_vec_kernel = lfilter(ker_hilbert, [1], sig_in_vec, axis=1)[:, :sig_in_vec.shape[1]]
    sig_in_vec_h = sig_in_vec + 1j * sig_in_vec_kernel

    cov_mat = sig_in_vec_h @ np.conjugate(sig_in_vec_h).T / sig_in_vec_h.shape[0]

    # compute the signal after beamforming
    sig_bf = bf_mat @ sig_in_vec_h

    power_bf = np.mean(np.abs(sig_bf) ** 2, axis=1)
    power_bf /= np.max(power_bf)

    plt.figure(figsize=(16, 10))
    plt.plot(angle_vec * 180 / np.pi, power_bf)
    plt.xlabel("angle [deg]")
    plt.ylabel("beamforming power")
    plt.title(f"beamforming when angle = {angle * 180 / np.pi:0.2f} deg")
    plt.grid(True)
    plt.show()


def test_hilbert_chirp_circular():
    """
    this function investigates the localization when the input signal is a chirp.
    This part targets circular arrays.
    """
    # array information
    speed = 340
    num_mic = 7
    freq_min = 1_000
    freq_max = 2_200

    lambda_min = speed / freq_max
    spacing = lambda_min / 2

    # compute the radius that achieves this spacing
    radius_arr = spacing / (2 * np.sin(np.pi / num_mic))
    radius_arr = 4.5e-2
    angle_arr = np.linspace(0, 2 * np.pi, num_mic)

    # signal information
    period = 1
    duration = 1.1 * period
    fs = 50_000
    time_vec = np.arange(0, duration, step=1 / fs)

    # angular grid
    num_grid = 16 * num_mic + 1
    angle_vec = np.linspace(-np.pi, np.pi, num_grid)

    # build the template for chirp freq and phase pattern
    # instantaneous freq
    freq_inst = freq_min + (freq_max - freq_min) * np.mod(time_vec, period) / period
    phase_inst = 2 * np.pi * np.cumsum(freq_inst) / fs

    ####################################################################################################################
    # Method 1: use Hilbert transform for all the signal
    ####################################################################################################################
    bf_mat = []

    for angle in tqdm(angle_vec):
        # produce the corresponding delays
        delay_vec = (radius_arr * np.cos(angle_arr - angle)) / speed
        delay_vec = delay_vec - np.min(delay_vec)

        # produce the signal
        sig_in_vec = []

        ph_init = 2 * np.pi * np.random.rand(1)[0]
        for delay in delay_vec:
            # delayed time
            time_delay = time_vec - delay
            time_delay[time_delay < 0] = 0

            # apply interpolation to recover the delayed phase
            phase = np.interp(time_delay, time_vec, phase_inst)
            sig_in = np.sin(phase + ph_init)

            sig_in_vec.append(sig_in)

        sig_in_vec = np.asarray(sig_in_vec)

        # compute the hilbert transform
        sig_in_vec_h = hilbert(sig_in_vec, axis=1)

        # compute the SVD for beamforming
        cov_mat = sig_in_vec_h @ np.conjugate(sig_in_vec_h).T / sig_in_vec_h.shape[0]
        U, D, _ = np.linalg.svd(cov_mat)

        # register the corresponding beamforming vector
        bf_mat.append(U[:, 0])

    # compute the array resolution
    bf_mat = np.conjugate(np.asarray(bf_mat))

    corr = bf_mat @ np.conjugate(bf_mat).T

    step = len(corr) // 4
    corr_selected = np.abs(corr[::step, :])

    plt.figure(figsize=(16, 10))
    plt.plot(angle_vec / np.pi * 180, corr_selected.T)
    plt.grid(True)
    plt.xlabel("angle [deg]")
    plt.ylabel("array resolution")
    plt.title(f"Hilbert: array resolution for a chirp in freq range [{freq_min:0.0f}, {freq_max:0.0f}] Hz")
    plt.ylim([0, 1.05])

    plt.show(block=False)

    ####################################################################################################################
    # Method 2: use short kernel representation of the Hilbert transform
    ####################################################################################################################

    # build the kernel
    ker_duration = 100e-3
    ker_len = int(ker_duration * fs)

    impulse = np.zeros(ker_len)
    impulse[0] = 1

    ker_hilbert = np.imag(hilbert(impulse))

    # build the beamforming matrices using the kernel
    bf_mat = []

    for angle in tqdm(angle_vec):
        # produce the corresponding delays
        delay_vec = (radius_arr * np.cos(angle_arr - angle)) / speed
        delay_vec = delay_vec - np.min(delay_vec)

        # produce the signal
        sig_in_vec = []

        ph_init = 2 * np.pi * np.random.rand(1)[0]
        for delay in delay_vec:
            # delayed time
            time_delay = time_vec - delay
            time_delay[time_delay < 0] = 0

            # apply interpolation to recover the delayed phase
            phase = np.interp(time_delay, time_vec, phase_inst)
            sig_in = np.sin(phase + ph_init)

            sig_in_vec.append(sig_in)

        sig_in_vec = np.asarray(sig_in_vec)

        # compute the hilbert transform
        sig_in_vec_kernel = lfilter(ker_hilbert, [1], sig_in_vec, axis=1)[:, :sig_in_vec.shape[1]]
        sig_in_vec_h = sig_in_vec + 1j * sig_in_vec_kernel

        # compute the SVD for beamforming
        cov_mat = sig_in_vec_h @ np.conjugate(sig_in_vec_h).T / sig_in_vec_h.shape[0]
        U, D, _ = np.linalg.svd(cov_mat)

        # register the corresponding beamforming vector
        bf_mat.append(U[:, 0])

    # compute the array resolution
    bf_mat = np.conjugate(np.asarray(bf_mat))

    corr = bf_mat @ np.conjugate(bf_mat).T

    step = len(corr) // 4
    corr_selected = np.abs(corr[::step, :])

    plt.figure(figsize=(16, 10))
    plt.plot(angle_vec / np.pi * 180, corr_selected.T)
    plt.grid(True)
    plt.xlabel("angle [deg]")
    plt.ylabel("array resolution")
    plt.title(
        f"Kernel Hilbert: array resolution for a chirp in freq range [{freq_min:0.0f}, {freq_max:0.0f}] Hz, ker-duartion: {ker_duration}, ker-len:{ker_len} samples")
    plt.ylim([0, 1.05])

    plt.show(block=False)

    # investigate the effect of beamforming in terms of amplitude
    angle = np.pi / 4

    # produce the corresponding delays
    delay_vec = (radius_arr * np.cos(angle_arr - angle)) / speed
    delay_vec = delay_vec - np.min(delay_vec)

    # produce the signal
    sig_in_vec = []

    ph_init = 2 * np.pi * np.random.rand(1)[0]
    for delay in delay_vec:
        # delayed time
        time_delay = time_vec - delay
        time_delay[time_delay < 0] = 0

        # apply interpolation to recover the delayed phase
        phase = np.interp(time_delay, time_vec, phase_inst)
        sig_in = np.sin(phase + ph_init)

        sig_in_vec.append(sig_in)

    sig_in_vec = np.asarray(sig_in_vec)

    # compute the hilbert transform
    sig_in_vec_kernel = lfilter(ker_hilbert, [1], sig_in_vec, axis=1)[:, :sig_in_vec.shape[1]]
    sig_in_vec_h = sig_in_vec + 1j * sig_in_vec_kernel

    cov_mat = sig_in_vec_h @ np.conjugate(sig_in_vec_h).T / sig_in_vec_h.shape[0]

    # compute the signal after beamforming
    sig_bf = bf_mat @ sig_in_vec_h

    power_bf = np.mean(np.abs(sig_bf) ** 2, axis=1)
    power_bf /= np.max(power_bf)

    plt.figure(figsize=(16, 10))
    # plt.plot(angle_vec * 180 / np.pi, power_bf)
    plt.axes(projection='polar')
    plt.polar(angle_vec, power_bf)
    plt.title(f"power profile for a target with DoA = {angle * 180 / np.pi:0.2f} deg")
    plt.grid(True)
    plt.show(block=False)

    ####################################################################################################################
    # Final step: investigate the effect of mismatch
    # NOTE: up to now we assume that the signal used for building beamforming vectors is the same as the test signal.
    # Here we would like to investigate if this is also true for other signals as well.
    ####################################################################################################################

    freq_probe = freq_max
    duration = 100e-3
    time_probe = np.arange(0, duration, step=1 / fs)

    angle = np.pi / 4
    delay_vec = (radius_arr * np.cos(angle_arr - angle)) / speed

    ph_init = 2 * np.pi * np.random.rand(1)[0]
    sig_in_vec = np.sin(2 * np.pi * freq_probe * (time_probe.reshape(1, -1) - delay_vec.reshape(-1, 1)) + ph_init)

    # compute the hilbert transform
    sig_in_vec_kernel = lfilter(ker_hilbert, [1], sig_in_vec, axis=1)[:, :sig_in_vec.shape[1]]
    sig_in_vec_h = sig_in_vec + 1j * sig_in_vec_kernel

    # compute the signal after beamforming
    sig_bf = bf_mat @ sig_in_vec_h

    power_bf = np.mean(np.abs(sig_bf) ** 2, axis=1)
    power_bf /= np.max(power_bf)

    plt.figure(figsize=(16, 10))
    # plt.plot(angle_vec * 180 / np.pi, power_bf)
    plt.axes(projection='polar')
    plt.polar(angle_vec, power_bf)
    plt.title(
        f"mismatch: power profile for a target with DoA = {angle * 180 / np.pi:0.2f} deg and sinusoid waveform with freq = {freq_probe:0.2f} Hz")
    plt.grid(True)
    plt.show(block=False)

    # repeat the simulation for a random signal
    cutoff = [freq_min/10, freq_max]
    order = 2
    b, a = butter(order, cutoff, 'bandpass', analog=False, output='ba', fs=fs)

    duration = 100e-3
    num_samples = int(duration * fs)

    noise = np.random.randn(num_samples)

    # waveform hitting the array
    time_vec = np.arange(0, duration, step=1 / fs)
    waveform = lfilter(b, a, noise)

    angle = np.pi/4
    delay_vec = (radius_arr * np.cos(angle_arr - angle))/speed

    delay_vec = delay_vec - np.min(delay_vec)

    sig_in_vec = []

    for delay in delay_vec:
        time_delay = time_vec - delay
        time_delay[time_delay < 0] = 0

        sig_in = np.interp(time_delay, time_vec, waveform)

        sig_in_vec.append(sig_in)

    sig_in_vec = np.asarray(sig_in_vec)

    # compute the hilbert transform
    sig_in_vec_kernel = lfilter(ker_hilbert, [1], sig_in_vec, axis=1)[:, :sig_in_vec.shape[1]]
    sig_in_vec_h = sig_in_vec + 1j * sig_in_vec_kernel

    # compute the signal after beamforming
    sig_bf = bf_mat @ sig_in_vec_h

    power_bf = np.mean(np.abs(sig_bf) ** 2, axis=1)
    power_bf /= np.max(power_bf)

    plt.figure(figsize=(16, 10))
    # plt.plot(angle_vec * 180 / np.pi, power_bf)

    # plt.subplot(211)
    plt.axes(projection='polar')
    plt.polar(angle_vec, power_bf)
    plt.title(
        f"mismatch: power profile for a target with DoA = {angle * 180 / np.pi:0.2f} deg and random waveform in freq range {cutoff} Hz")
    plt.grid(True)

    # plt.subplot(212)
    # plt.plot(time_vec, waveform)
    # plt.xlabel("time (sec)")
    # plt.ylabel("waveform")
    plt.show()





def test_hilbert_freq_domain():
    """
    this module investigates the frequency domain representation of the Hilbert kernel.
    """
    # kernel length
    L = 100

    impulse = np.zeros(L)
    impulse[0] = 1

    ker_h = np.imag(hilbert(impulse))

    # plot the frequency domain representation of the filter
    w, h = freqz(ker_h, [1], worN=10 * L)
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.grid(True)
    plt.xlabel("frequency")
    plt.ylabel("frequency response")
    plt.show()


def main():
    # test_hilbert_SNN()
    # test_hilbert_harmonics()
    # test_hilbert_SNN_localization()
    # test_hilbert_SNN_beamforming()
    # test_hilbert_kernel()
    # test_short_hilbert()
    # test_hilbert_chirp()
    test_hilbert_chirp_circular()
    # test_hilbert_freq_domain()


if __name__ == '__main__':
    main()
