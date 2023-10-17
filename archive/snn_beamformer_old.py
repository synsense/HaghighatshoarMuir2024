# ----------------------------------------------------------------------------------------------------------------------
# This module allows to design beamforming matrices for SNN networks.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 12.07.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from micloc.array_geometry import ArrayGeometry
from micloc.spike_encoder import IAFSpikeEncoder, ZeroCrossingSpikeEncoder, IAFZeroCrossingSpikeEncoder, \
    PeakSpikeEncoder
from typing import Tuple, Union
from scipy.signal import hilbert, lfilter, butter
from numbers import Number
from tqdm import tqdm
import matplotlib.pyplot as plt

# FIXME: the code needs some check: sometimes we apply to filtered version sometimes not? also we need to clean up the approach
# FIXME: we need to make sure that the real version is not sensitive to time shift. In general, it can but I think in STHT version it is not since I and Q are almost orthogonal to each orther

class SNNBeamformer:
    def __init__(self, geometry: ArrayGeometry, kernel_duration: np.ndarray, tau_vec: np.ndarray,
                 target_spike_rate: float, fs: float):
        """
        this class builds methods for building beamforming matrices in multi-mic arrays when the input is spiky.
        Args:
            geometry (ArrayGeometry): encoding of the geometry of the array in terms of time-of-arrivals.
            kernel_duration (float): duration of the Hilbert kernel applied to the spikes.
            tau_vec (np.ndarray): a list consisting of tau-syn and tau_mem of the neuron.
            target_spike_rate (float): desired spike rate at which the beamformers are designed.
            fs (float): sampling rate of the array.
        """
        self.geometry = geometry
        self.fs = fs

        # build the corresponding kernel
        self.kernel_duration = kernel_duration
        self.kernel_length = int(self.fs * self.kernel_duration)
        impulse = np.zeros(self.kernel_length)
        impulse[0] = 1
        self.kernel = np.fft.fftshift(np.imag(hilbert(impulse)))

        # build the corresponding neuron kernel
        self.tau_vec = tau_vec

        self.target_spike_rate = target_spike_rate

        # self.spk_encoder = IAFSpikeEncoder(target_spike_rate=target_spike_rate, fs=self.fs)
        self.spk_encoder = ZeroCrossingSpikeEncoder(fs=self.fs, robust_width=100)
        # self.spk_encoder = IAFZeroCrossingSpikeEncoder(target_spike_rate=target_spike_rate, fs=self.fs)
        # self.spk_encoder = PeakSpikeEncoder(fs=self.fs)

        # generate the bandpass filter for the spikes
        # NOTE: this is needed since in STHT we have a imitation on the frequency from below
        f_low = 4 / self.kernel_duration
        f_max = target_spike_rate
        order = 2
        cutoff = [f_low, f_max]

        self.bandpass_filter = butter(order, cutoff, btype="bandpass", analog=False, output="ba", fs=fs)

    def design_from_template(self, template: Tuple[np.ndarray, np.ndarray], doa_list: np.ndarray) -> np.ndarray:
        """
        this function builds suitable beamforming matrices for the array when it receives a given waveform in spiky format.
        Args:
            template (Tuple[np.ndarray, np.ndarray]): a tuple containing the template signal (time_temp, sig_temp).
            doa_list (np.ndarray): a list containing target DoAs for which the beamforming is designed.

        Returns:
            a matrix containing beamforming vectors for various DoAs.

        """
        # extract the signal
        try:
            time_temp, sig_temp = template
        except Exception:
            raise ValueError("input template should be a tuple containing (time_in, sig_in) of the template signal!")

        # resample the signal to the clock rate of the array
        time_interp = np.arange(time_temp.min(), time_temp.max(), step=1 / self.fs)
        sig_interp = np.interp(time_interp, time_temp, sig_temp)

        sig_temp, time_temp = sig_interp, time_interp

        # matrix containing dominant directions in its columns
        bf_mat = []

        print()
        print('+' * 150)
        print(" designing SNN beamforming matrices for various DoAs ".center(150, '+'))
        print('+' * 150)

        # build the neuron kernel
        tau_syn, tau_mem = self.tau_vec[0], self.tau_vec[1]
        time_neuron = time_temp - time_temp[0]

        if tau_mem == tau_syn:
            neuron_impulse_response = (time_neuron / tau_syn) * np.exp(-time_neuron / tau_syn)
        else:
            neuron_impulse_response = (np.exp(-time_neuron / tau_syn) - np.exp(time_neuron / tau_mem)) / (
                    1 / tau_mem - 1 / tau_syn)
            assert np.all(neuron_impulse_response >= 0)

        # normalize the impulse response
        neuron_impulse_response = neuron_impulse_response / np.sum(neuron_impulse_response)
        effective_length = np.sum(np.cumsum(neuron_impulse_response) < 0.999)

        neuron_impulse_response = neuron_impulse_response[:effective_length]

        for doa in tqdm(doa_list):
            # compute the delays associated with the DoA
            delays = self.geometry.delays(
                theta=doa,
                normalized=True
            )

            # interpolate the template signal to obtain the incoming signal
            delays -= delays.min()

            time_delayed = time_temp.reshape(1, -1) - delays.reshape(-1, 1)
            time_delayed[time_delayed < time_temp.min()] = time_temp.min()

            sig_in_vec = np.interp(time_delayed.ravel(), time_temp, sig_temp).reshape(time_delayed.shape)

            # convert the input signal into spikes
            spikes_in_vec = self.spk_encoder.evolve(sig_in_vec.T)

            # remove the low-pass part of the signal so that STHT works well
            b, a = self.bandpass_filter
            spikes_in_vec = lfilter(b, a, spikes_in_vec, axis=0)

            # compute the in-phase and quadrature parts before applying to the neuron
            sig_in_vec_h = np.roll(spikes_in_vec, self.kernel_length // 2, axis=0) + 1j * lfilter(self.kernel, [1],
                                                                                                  spikes_in_vec, axis=0)

            # compute the filtered version
            sig_in_vec_h_filtered = lfilter(neuron_impulse_response, [1], sig_in_vec_h, axis=0)
            # sig_in_vec_h_filtered = sig_in_vec_h

            # now that the input signals in all arrays are available, design the beamformer
            # 1. remove the transient part
            stable_part = sig_in_vec_h_filtered.shape[0] // 4
            sig_in_vec_h_filtered_stable = sig_in_vec_h_filtered[stable_part:, :]

            phase = np.unwrap(np.angle(sig_in_vec_h_filtered_stable), axis=0)

            # 2. compute the covariance matrix
            cov_mat = 1 / sig_in_vec_h_filtered_stable.shape[0] * (sig_in_vec_h_filtered_stable.T @ sig_in_vec_h_filtered_stable.conj())

            U, D, _ = np.linalg.svd(cov_mat)

            bf_mat.append(U[:, 0])

        bf_mat = np.asarray(bf_mat).T

        return bf_mat

    def apply_to_template(self, bf_mat: np.ndarray, template: Tuple[np.ndarray, np.ndarray, Union[Number, np.ndarray]],
                          snr_db: float) -> np.ndarray:
        """
        this module applies beamforing when the array receives a given template [time_temp, sig_temp, doa_temp] and returns the signal after beamforming.
        Notes: in this version, the DoA of the source transmitting the template signal may vary during time.

        Args:
            bf_mat (np.ndarray): matrix containing beamforming vectors at various angles.
            template Tuple[np.ndarray, np.ndarray, Union[Number, np.ndarray]]): input template signal consisting of time-of-arrival, signal samples and direction of arrival.
            snr_db (float): signal to noise ratio at each array element.

        Returns:
            signal after beamforming.
        """
        # extract the signal
        try:
            time_temp, sig_temp, doa_temp = template
        except Exception:
            raise ValueError(
                "input template should be a tuple containing (time_in, sig_in, doa_in) of the template signal!")

        if isinstance(doa_temp, Number):
            doa_temp = doa_temp * np.ones_like(sig_temp)

        # compute SNR
        snr = 10 ** (snr_db / 10)

        # resample the signal to the clock rate of the array
        time_in = np.arange(time_temp.min(), time_temp.max(), step=1 / self.fs)
        sig_in = np.interp(time_in, time_temp, sig_temp)
        doa_in = np.interp(time_in, time_temp, doa_temp)

        time_temp, sig_temp, doa_temp = time_in, sig_in, doa_in

        # compute the signal received at the array
        # NOTE: here we apply a delay normalization to all the samples later rather than sample by sample normalization which yields wrong results.
        delays = np.asarray([self.geometry.delays(theta=doa, normalized=False) for doa in doa_temp]).T
        delays = delays - delays.min()

        time_delayed = time_temp.reshape(1, -1) - delays
        time_delayed[time_delayed < time_temp.min()] = time_temp.min()

        sig_in_vec = np.interp(time_delayed.ravel(), time_temp, sig_temp).reshape(time_delayed.shape)

        # add noise to the received signal in the array
        noise = np.sqrt(np.mean(sig_in_vec ** 2)) / np.sqrt(snr) * np.random.randn(*sig_in_vec.shape)
        sig_in_vec += noise

        # convert the input signal into spikes
        spikes_in_vec = self.spk_encoder.evolve(sig_in_vec.T)

        # apply bandpass filtering to get rid of the low-pass part of the signal
        b, a = self.bandpass_filter
        spikes_in_vec = lfilter(b, a, spikes_in_vec, axis=0)

        # compute the in-phase and quadrature parts before applying to the neuron
        sig_in_vec_h = np.roll(spikes_in_vec, self.kernel_length // 2, axis=0) + 1j * lfilter(self.kernel, [1],
                                                                                              spikes_in_vec, axis=0)

        ## compute the filtered version after being processed by the neuron
        # build the neuron kernel
        tau_syn, tau_mem = self.tau_vec[0], self.tau_vec[1]
        time_neuron = time_temp - time_temp[0]

        if tau_mem == tau_syn:
            neuron_impulse_response = (time_neuron / tau_syn) * np.exp(-time_neuron / tau_syn)
        else:
            neuron_impulse_response = (np.exp(-time_neuron / tau_syn) - np.exp(time_neuron / tau_mem)) / (
                    1 / tau_mem - 1 / tau_syn)
            assert np.all(neuron_impulse_response >= 0)

        # normalize the impulse response
        neuron_impulse_response = neuron_impulse_response / np.sum(neuron_impulse_response)
        effective_length = np.sum(np.cumsum(neuron_impulse_response) < 0.999)

        neuron_impulse_response = neuron_impulse_response[:effective_length]

        sig_in_vec_h_filtered = lfilter(neuron_impulse_response, [1], sig_in_vec_h, axis=0)

        # apply the beamforming
        sig_in_beamformed = bf_mat.conj().T @ sig_in_vec_h.T

        return sig_in_beamformed

    def apply_to_signal(self, bf_mat: np.ndarray, data_in: np.ndarray) -> np.ndarray:
        """
        this function applies bemforming for a given signal received from array elements.
        Args:
            bf_mat (np.ndarray): beamforming matrix.
            data_in (np.ndarray): input data.

        Returns:
            signal after beamforming.
        """

        num_mic, num_grid = bf_mat.shape

        num_chan, T = data_in.shape

        if num_chan != num_mic:
            raise ValueError(
                f"number of channels in the input siganl {num_chan} should be the same as the number of microphones {num_mic}!")

        # convert the input signal into spikes
        spikes_in_vec = self.spk_encoder.evolve(data_in.T)

        # convert spikes into +1 and -1 for better phase stability
        spikes_in_vec = 2 * spikes_in_vec - 1

        # compute the kernel Hilbert transform of the input signal
        data_in_kernel_H = spikes_in_vec + 1j * lfilter(self.kernel, [1], spikes_in_vec, axis=1)

        # apply bemaforming
        data_bf = bf_mat.conj().T @ data_in_kernel_H

        return data_bf


class SNNBeamformerReal:
    def __init__(self, geometry: ArrayGeometry, kernel_duration: np.ndarray, tau_vec: np.ndarray,
                 target_spike_rate: float, fs: float):
        """
        this class builds methods for building beamforming matrices in multi-mic arrays when the input is spiky.
        Args:
            geometry (ArrayGeometry): encoding of the geometry of the array in terms of time-of-arrivals.
            kernel_duration (float): duration of the Hilbert kernel applied to the spikes.
            tau_vec (np.ndarray): a list consisting of tau-syn and tau_mem of the neuron.
            target_spike_rate (float): desired spike rate at which the beamformers are designed.
            fs (float): sampling rate of the array.
        """
        self.geometry = geometry
        self.fs = fs

        # build the corresponding kernel
        self.kernel_duration = kernel_duration
        self.kernel_length = int(self.fs * self.kernel_duration)
        impulse = np.zeros(self.kernel_length)
        impulse[0] = 1
        self.kernel = np.fft.fftshift(np.imag(hilbert(impulse)))

        # build the corresponding neuron kernel
        self.tau_vec = tau_vec

        self.target_spike_rate = target_spike_rate

        # self.spk_encoder = IAFSpikeEncoder(target_spike_rate=target_spike_rate, fs=self.fs)
        self.spk_encoder = ZeroCrossingSpikeEncoder(fs=self.fs, robust_width=100)
        # self.spk_encoder = IAFZeroCrossingSpikeEncoder(target_spike_rate=target_spike_rate, fs=self.fs)
        # self.spk_encoder = PeakSpikeEncoder(fs=self.fs)

        # generate the bandpass filter for the spikes
        # NOTE: this is needed since in STHT we have a imitation on the frequency from below
        f_low = 4 / self.kernel_duration
        f_max = target_spike_rate
        order = 2
        cutoff = [f_low, f_max]

        self.bandpass_filter = butter(order, cutoff, btype="bandpass", analog=False, output="ba", fs=fs)

    def design_from_template(self, template: Tuple[np.ndarray, np.ndarray], doa_list: np.ndarray) -> np.ndarray:
        """
        this function builds suitable beamforming matrices for the array when it receives a given waveform in spiky format.
        Args:
            template (Tuple[np.ndarray, np.ndarray]): a tuple containing the template signal (time_temp, sig_temp).
            doa_list (np.ndarray): a list containing target DoAs for which the beamforming is designed.

        Returns:
            a matrix containing beamforming vectors for various DoAs.

        """
        # extract the signal
        try:
            time_temp, sig_temp = template
        except Exception:
            raise ValueError("input template should be a tuple containing (time_in, sig_in) of the template signal!")

        # resample the signal to the clock rate of the array
        time_interp = np.arange(time_temp.min(), time_temp.max(), step=1 / self.fs)
        sig_interp = np.interp(time_interp, time_temp, sig_temp)

        sig_temp, time_temp = sig_interp, time_interp

        # matrix containing dominant directions in its columns
        bf_mat = []

        print()
        print('+' * 150)
        print(" designing SNN beamforming matrices for various DoAs ".center(150, '+'))
        print('+' * 150)

        # build the neuron kernel
        tau_syn, tau_mem = self.tau_vec[0], self.tau_vec[1]
        time_neuron = time_temp - time_temp[0]

        if tau_mem == tau_syn:
            neuron_impulse_response = (time_neuron / tau_syn) * np.exp(-time_neuron / tau_syn)
        else:
            neuron_impulse_response = (np.exp(-time_neuron / tau_syn) - np.exp(time_neuron / tau_mem)) / (
                    1 / tau_mem - 1 / tau_syn)
            assert np.all(neuron_impulse_response >= 0)

        # normalize the impulse response
        neuron_impulse_response = neuron_impulse_response / np.sum(neuron_impulse_response)
        effective_length = np.sum(np.cumsum(neuron_impulse_response) < 0.999)

        neuron_impulse_response = neuron_impulse_response[:effective_length]

        for doa in tqdm(doa_list):
            # compute the delays associated with the DoA
            delays = self.geometry.delays(
                theta=doa,
                normalized=True
            )

            # interpolate the template signal to obtain the incoming signal
            delays -= delays.min()

            time_delayed = time_temp.reshape(1, -1) - delays.reshape(-1, 1)
            time_delayed[time_delayed < time_temp.min()] = time_temp.min()

            sig_in_vec = np.interp(time_delayed.ravel(), time_temp, sig_temp).reshape(time_delayed.shape)

            # convert the input signal into spikes
            spikes_in_vec = self.spk_encoder.evolve(sig_in_vec.T)

            # remove the low-pass part of the signal so that STHT works well
            b, a = self.bandpass_filter
            spikes_in_vec = lfilter(b, a, spikes_in_vec, axis=0)

            # compute the in-phase and quadrature parts before applying to the neuron
            sig_in_vec_h = np.roll(spikes_in_vec, self.kernel_length // 2, axis=0) + 1j * lfilter(self.kernel, [1],
                                                                                                  spikes_in_vec, axis=0)

            # compute the filtered version
            sig_in_vec_h_filtered = lfilter(neuron_impulse_response, [1], sig_in_vec_h, axis=0)
            # sig_in_vec_h_filtered = sig_in_vec_h

            # now that the input signals in all arrays are available, design the beamformer
            # 1. remove the transient part
            stable_part = sig_in_vec_h_filtered.shape[0] // 4
            sig_in_vec_h_filtered_stable = sig_in_vec_h_filtered[stable_part:, :]

            phase = np.unwrap(np.angle(sig_in_vec_h_filtered_stable), axis=0)

            # 2. compute the covariance matrix
            sig_real = np.hstack([np.real(sig_in_vec_h_filtered_stable), np.imag(sig_in_vec_h_filtered_stable)])
            cov_mat = 1 / sig_real.shape[0] * (sig_real.T @ sig_real)

            U, D, _ = np.linalg.svd(cov_mat)

            bf_mat.append(U[:, 0])

        bf_mat = np.asarray(bf_mat).T

        return bf_mat

    def apply_to_template(self, bf_mat: np.ndarray, template: Tuple[np.ndarray, np.ndarray, Union[Number, np.ndarray]],
                          snr_db: float) -> np.ndarray:
        """
        this module applies beamforing when the array receives a given template [time_temp, sig_temp, doa_temp] and returns the signal after beamforming.
        Notes: in this version, the DoA of the source transmitting the template signal may vary during time.

        Args:
            bf_mat (np.ndarray): matrix containing beamforming vectors at various angles.
            template Tuple[np.ndarray, np.ndarray, Union[Number, np.ndarray]]): input template signal consisting of time-of-arrival, signal samples and direction of arrival.
            snr_db (float): signal to noise ratio at each array element.

        Returns:
            signal after beamforming.
        """
        # extract the signal
        try:
            time_temp, sig_temp, doa_temp = template
        except Exception:
            raise ValueError(
                "input template should be a tuple containing (time_in, sig_in, doa_in) of the template signal!")

        if isinstance(doa_temp, Number):
            doa_temp = doa_temp * np.ones_like(sig_temp)

        # compute SNR
        snr = 10 ** (snr_db / 10)

        # resample the signal to the clock rate of the array
        time_in = np.arange(time_temp.min(), time_temp.max(), step=1 / self.fs)
        sig_in = np.interp(time_in, time_temp, sig_temp)
        doa_in = np.interp(time_in, time_temp, doa_temp)

        time_temp, sig_temp, doa_temp = time_in, sig_in, doa_in

        # compute the signal received at the array
        # NOTE: here we apply a delay normalization to all the samples later rather than sample by sample normalization which yields wrong results.
        delays = np.asarray([self.geometry.delays(theta=doa, normalized=False) for doa in doa_temp]).T
        delays = delays - delays.min()

        time_delayed = time_temp.reshape(1, -1) - delays
        time_delayed[time_delayed < time_temp.min()] = time_temp.min()

        sig_in_vec = np.interp(time_delayed.ravel(), time_temp, sig_temp).reshape(time_delayed.shape)

        # add noise to the received signal in the array
        noise = np.sqrt(np.mean(sig_in_vec ** 2)) / np.sqrt(snr) * np.random.randn(*sig_in_vec.shape)
        sig_in_vec += noise

        # convert the input signal into spikes
        spikes_in_vec = self.spk_encoder.evolve(sig_in_vec.T)

        # apply bandpass filtering to get rid of the low-pass part of the signal
        b, a = self.bandpass_filter
        spikes_in_vec = lfilter(b, a, spikes_in_vec, axis=0)

        # compute the in-phase and quadrature parts before applying to the neuron
        sig_in_vec_h = np.roll(spikes_in_vec, self.kernel_length // 2, axis=0) + 1j * lfilter(self.kernel, [1],
                                                                                              spikes_in_vec, axis=0)

        ## compute the filtered version after being processed by the neuron
        # build the neuron kernel
        tau_syn, tau_mem = self.tau_vec[0], self.tau_vec[1]
        time_neuron = time_temp - time_temp[0]

        if tau_mem == tau_syn:
            neuron_impulse_response = (time_neuron / tau_syn) * np.exp(-time_neuron / tau_syn)
        else:
            neuron_impulse_response = (np.exp(-time_neuron / tau_syn) - np.exp(time_neuron / tau_mem)) / (
                    1 / tau_mem - 1 / tau_syn)
            assert np.all(neuron_impulse_response >= 0)

        # normalize the impulse response
        neuron_impulse_response = neuron_impulse_response / np.sum(neuron_impulse_response)
        effective_length = np.sum(np.cumsum(neuron_impulse_response) < 0.999)

        neuron_impulse_response = neuron_impulse_response[:effective_length]

        sig_in_vec_h_filtered = lfilter(neuron_impulse_response, [1], sig_in_vec_h, axis=0)

        # apply the beamforming
        sig_real = np.hstack([np.real(sig_in_vec_h_filtered), np.imag(sig_in_vec_h_filtered)])
        sig_in_beamformed = bf_mat.T @ sig_real.T

        return sig_in_beamformed

    def apply_to_signal(self, bf_mat: np.ndarray, data_in: np.ndarray) -> np.ndarray:
        """
        this function applies beamforming for a given signal received from array elements.
        Args:
            bf_mat (np.ndarray): beamforming matrix.
            data_in (np.ndarray): input data.

        Returns:
            signal after beamforming.
        """

        num_mic, num_grid = bf_mat.shape

        num_chan, T = data_in.shape

        if num_chan != num_mic:
            raise ValueError(
                f"number of channels in the input siganl {num_chan} should be the same as the number of microphones {num_mic}!")

        # convert the input signal into spikes
        spikes_in_vec = self.spk_encoder.evolve(data_in.T)

        # convert spikes into +1 and -1 for better phase stability
        spikes_in_vec = 2 * spikes_in_vec - 1

        # compute the kernel Hilbert transform of the input signal
        data_in_kernel_H = spikes_in_vec + 1j * lfilter(self.kernel, [1], spikes_in_vec, axis=1)

        # apply bemaforming
        data_bf = bf_mat.conj().T @ data_in_kernel_H

        return data_bf



class SNNBeamformerSpiky:
    def __init__(self, geometry: ArrayGeometry, kernel_duration: np.ndarray, tau_vec: np.ndarray,
                 target_spike_rate: float, fs: float):
        """
        this class builds methods for building beamforming matrices in multi-mic arrays when the input is spiky.
        Args:
            geometry (ArrayGeometry): encoding of the geometry of the array in terms of time-of-arrivals.
            kernel_duration (float): duration of the Hilbert kernel applied to the spikes.
            tau_vec (np.ndarray): a list consisting of tau-syn and tau_mem of the neuron.
            target_spike_rate (float): desired spike rate at which the beamformers are designed.
            fs (float): sampling rate of the array.
        """
        self.geometry = geometry
        self.fs = fs

        # build the corresponding kernel
        self.kernel_duration = kernel_duration
        self.kernel_length = int(self.fs * self.kernel_duration)
        impulse = np.zeros(self.kernel_length)
        impulse[0] = 1
        self.kernel = np.fft.fftshift(np.imag(hilbert(impulse)))

        # build the corresponding neuron kernel
        self.tau_vec = tau_vec

        self.target_spike_rate = target_spike_rate

        # self.spk_encoder = IAFSpikeEncoder(target_spike_rate=target_spike_rate, fs=self.fs)
        self.spk_encoder = ZeroCrossingSpikeEncoder(fs=self.fs, robust_width=15)
        # self.spk_encoder = IAFSpikeEncoder(target_spike_rate=target_spike_rate, fs=fs)
        # self.spk_encoder = IAFZeroCrossingSpikeEncoder(target_spike_rate=target_spike_rate, fs=self.fs)
        # self.spk_encoder = PeakSpikeEncoder(fs=self.fs)

        # generate the bandpass filter for the spikes
        # NOTE: this is needed since in STHT we have a imitation on the frequency from below
        f_low = 4 / (2*np.pi*self.kernel_duration)
        f_max = 2_000 #target_spike_rate
        order = 2
        cutoff = [1500, f_max]

        self.bandpass_filter = butter(order, cutoff, btype="bandpass", analog=False, output="ba", fs=fs)

    def design_from_template(self, template: Tuple[np.ndarray, np.ndarray], doa_list: np.ndarray) -> np.ndarray:
        """
        this function builds suitable beamforming matrices for the array when it receives a given waveform in spiky format.
        Args:
            template (Tuple[np.ndarray, np.ndarray]): a tuple containing the template signal (time_temp, sig_temp).
            doa_list (np.ndarray): a list containing target DoAs for which the beamforming is designed.

        Returns:
            a matrix containing beamforming vectors for various DoAs.

        """
        # extract the signal
        try:
            time_temp, sig_temp = template
        except Exception:
            raise ValueError("input template should be a tuple containing (time_in, sig_in) of the template signal!")

        # resample the signal to the clock rate of the array
        time_interp = np.arange(time_temp.min(), time_temp.max(), step=1 / self.fs)
        sig_interp = np.interp(time_interp, time_temp, sig_temp)

        sig_temp, time_temp = sig_interp, time_interp

        # matrix containing dominant directions in its columns
        bf_mat = []

        print()
        print('+' * 150)
        print(" designing SNN beamforming matrices for various DoAs ".center(150, '+'))
        print('+' * 150)

        # build the neuron kernel
        tau_syn, tau_mem = self.tau_vec[0], self.tau_vec[1]
        time_neuron = time_temp - time_temp[0]

        if tau_mem == tau_syn:
            neuron_impulse_response = (time_neuron / tau_syn) * np.exp(-time_neuron / tau_syn)
        else:
            neuron_impulse_response = (np.exp(-time_neuron / tau_syn) - np.exp(time_neuron / tau_mem)) / (
                    1 / tau_mem - 1 / tau_syn)
            assert np.all(neuron_impulse_response >= 0)

        # normalize the impulse response
        neuron_impulse_response = neuron_impulse_response / np.sum(neuron_impulse_response)
        effective_length = np.sum(np.cumsum(neuron_impulse_response) < 0.999)

        neuron_impulse_response = neuron_impulse_response[:effective_length]

        for doa in tqdm(doa_list):
            # compute the delays associated with the DoA
            delays = self.geometry.delays(
                theta=doa,
                normalized=True
            )

            # interpolate the template signal to obtain the incoming signal
            delays -= delays.min()

            time_delayed = time_temp.reshape(1, -1) - delays.reshape(-1, 1)
            time_delayed[time_delayed < time_temp.min()] = time_temp.min()

            sig_in_vec = np.interp(time_delayed.ravel(), time_temp, sig_temp).reshape(time_delayed.shape).T

            # compute the in-phase and quadrature parts before applying to the neuron
            sig_in_vec_h = np.roll(sig_in_vec, self.kernel_length // 2, axis=0) + 1j * lfilter(self.kernel, [1], sig_in_vec, axis=0)

            # remove the low-pass part of the signal so that STHT works well
            b, a = self.bandpass_filter
            sig_in_vec_h_bp = lfilter(b, a, sig_in_vec_h, axis=0)

            # obtain the real-valued version of the signal
            sig_in_real = np.hstack([np.real(sig_in_vec_h_bp), np.imag(sig_in_vec_h_bp)])

            # obtain the spike encoding
            spikes_vec = self.spk_encoder.evolve(sig_in_real)

            # compute the filtered version
            vmem_vec = lfilter(neuron_impulse_response, [1], spikes_vec, axis=0)
            # sig_in_vec_h_filtered = sig_in_vec_h

            # now that the input signals in all arrays are available, design the beamformer
            # 1. remove the transient part
            stable_part = vmem_vec.shape[0] // 4
            vmem_stable = vmem_vec[stable_part:, :]

            vmem_stable -= vmem_stable.mean()

            # 2. compute the covariance matrix
            cov_mat = 1 / vmem_stable.shape[0] * (vmem_stable.T @ vmem_stable)

            U, D, _ = np.linalg.svd(cov_mat)

            bf_mat.append(U[:, 0])

        bf_mat = np.asarray(bf_mat).T

        return bf_mat

    def apply_to_template(self, bf_mat: np.ndarray, template: Tuple[np.ndarray, np.ndarray, Union[Number, np.ndarray]],
                          snr_db: float) -> np.ndarray:
        """
        this module applies beamforing when the array receives a given template [time_temp, sig_temp, doa_temp] and returns the signal after beamforming.
        Notes: in this version, the DoA of the source transmitting the template signal may vary during time.

        Args:
            bf_mat (np.ndarray): matrix containing beamforming vectors at various angles.
            template Tuple[np.ndarray, np.ndarray, Union[Number, np.ndarray]]): input template signal consisting of time-of-arrival, signal samples and direction of arrival.
            snr_db (float): signal to noise ratio at each array element.

        Returns:
            signal after beamforming.
        """
        # extract the signal
        try:
            time_temp, sig_temp, doa_temp = template
        except Exception:
            raise ValueError(
                "input template should be a tuple containing (time_in, sig_in, doa_in) of the template signal!")

        if isinstance(doa_temp, Number):
            doa_temp = doa_temp * np.ones_like(sig_temp)

        # compute SNR
        snr = 10 ** (snr_db / 10)

        # resample the signal to the clock rate of the array
        time_in = np.arange(time_temp.min(), time_temp.max(), step=1 / self.fs)
        sig_in = np.interp(time_in, time_temp, sig_temp)
        doa_in = np.interp(time_in, time_temp, doa_temp)

        time_temp, sig_temp, doa_temp = time_in, sig_in, doa_in

        # compute the signal received at the array
        # NOTE: here we apply a delay normalization to all the samples later rather than sample by sample normalization which yields wrong results.
        delays = np.asarray([self.geometry.delays(theta=doa, normalized=False) for doa in doa_temp]).T
        delays = delays - delays.min()

        time_delayed = time_temp.reshape(1, -1) - delays
        time_delayed[time_delayed < time_temp.min()] = time_temp.min()

        sig_in_vec = np.interp(time_delayed.ravel(), time_temp, sig_temp).reshape(time_delayed.shape)

        # add noise to the received signal in the array
        noise = np.sqrt(np.mean(sig_in_vec ** 2)) / np.sqrt(snr) * np.random.randn(*sig_in_vec.shape)
        sig_in_vec += noise

        # convert the input signal into spikes
        spikes_in_vec = self.spk_encoder.evolve(sig_in_vec.T)

        # apply bandpass filtering to get rid of the low-pass part of the signal
        b, a = self.bandpass_filter
        spikes_in_vec = lfilter(b, a, spikes_in_vec, axis=0)

        # compute the in-phase and quadrature parts before applying to the neuron
        sig_in_vec_h = np.roll(spikes_in_vec, self.kernel_length // 2, axis=0) + 1j * lfilter(self.kernel, [1],
                                                                                              spikes_in_vec, axis=0)

        ## compute the filtered version after being processed by the neuron
        # build the neuron kernel
        tau_syn, tau_mem = self.tau_vec[0], self.tau_vec[1]
        time_neuron = time_temp - time_temp[0]

        if tau_mem == tau_syn:
            neuron_impulse_response = (time_neuron / tau_syn) * np.exp(-time_neuron / tau_syn)
        else:
            neuron_impulse_response = (np.exp(-time_neuron / tau_syn) - np.exp(time_neuron / tau_mem)) / (
                    1 / tau_mem - 1 / tau_syn)
            assert np.all(neuron_impulse_response >= 0)

        # normalize the impulse response
        neuron_impulse_response = neuron_impulse_response / np.sum(neuron_impulse_response)
        effective_length = np.sum(np.cumsum(neuron_impulse_response) < 0.999)

        neuron_impulse_response = neuron_impulse_response[:effective_length]

        sig_in_vec_h_filtered = lfilter(neuron_impulse_response, [1], sig_in_vec_h, axis=0)

        # apply the beamforming
        sig_real = np.hstack([np.real(sig_in_vec_h_filtered), np.imag(sig_in_vec_h_filtered)])
        sig_in_beamformed = bf_mat.T @ sig_real.T

        return sig_in_beamformed

    def apply_to_signal(self, bf_mat: np.ndarray, data_in: np.ndarray) -> np.ndarray:
        """
        this function applies beamforming for a given signal received from array elements.
        Args:
            bf_mat (np.ndarray): beamforming matrix.
            data_in (np.ndarray): input data.

        Returns:
            signal after beamforming.
        """

        num_mic, num_grid = bf_mat.shape

        num_chan, T = data_in.shape

        if num_chan != num_mic:
            raise ValueError(
                f"number of channels in the input siganl {num_chan} should be the same as the number of microphones {num_mic}!")

        # convert the input signal into spikes
        spikes_in_vec = self.spk_encoder.evolve(data_in.T)

        # convert spikes into +1 and -1 for better phase stability
        spikes_in_vec = 2 * spikes_in_vec - 1

        # compute the kernel Hilbert transform of the input signal
        data_in_kernel_H = spikes_in_vec + 1j * lfilter(self.kernel, [1], spikes_in_vec, axis=1)

        # apply bemaforming
        data_bf = bf_mat.conj().T @ data_in_kernel_H

        return data_bf