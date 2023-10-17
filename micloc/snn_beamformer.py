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
from micloc.spike_encoder import ZeroCrossingSpikeEncoder
from typing import Tuple, Union
from scipy.signal import hilbert, lfilter, butter
from numbers import Number
from tqdm import tqdm
import matplotlib.pyplot as plt

# sampling rate of multi-mic board
Fs = 48_000

class SNNBeamformer:
    def __init__(self, geometry: ArrayGeometry, kernel_duration: np.ndarray, freq_range: np.ndarray,
                 tau_vec: np.ndarray, bipolar_spikes: bool = False,
                 fs: float = Fs):
        """
        this class implements an algorithm for building beamforming matrices in multi-mic arrays when the input is spiky.
        Args:
            geometry (ArrayGeometry): encoding of the geometry of the array in terms of time-of-arrivals.
            kernel_duration (float): duration of the Hilbert kernel applied to the spikes.
            freq_range (np.ndarray): an array containing f_low and f_high frequencies used for bandpass filtering.
            NOTE: this is needed to make sure that the STHT matches very well the original Hilbert transform.
            tau_vec (np.ndarray): a list consisting of tau-syn and tau_mem of the neuron.
            bipolar_spikes (bool): if spikes should be bipolar +1/-1 or unipolar +1. Defaults to unipolar +1.
            fs (float): sampling rate of the array. Defaults to Fs=48K in multi-mic board.
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

        # generate the bandpass filter used to make sure that STHT matches the original Hilbert transform
        try:
            f_low, f_high = freq_range
            if f_low > f_high:
                raise Exception()
        except Exception:
            raise ValueError("freq_range should be a vector consisting of two frequencies f_low < f_high!")

        order = 2
        cutoff = freq_range
        self.bandpass_filter = butter(order, cutoff, btype="bandpass", analog=False, output="ba", fs=fs)

        # distance between two consecutive zero crossing
        zc_dist = int(fs / f_high)
        robust_width = zc_dist // 2
        self.bipolar_spikes = bipolar_spikes
        self.spk_encoder = ZeroCrossingSpikeEncoder(fs=self.fs, robust_width=robust_width, bipolar=bipolar_spikes)

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

        # beamforming matrices obtained using SVD applied to the signal covariance matrix
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

            # signal after interpolation of dim: `T x num_chan`
            sig_in_vec = np.interp(time_delayed.ravel(), time_temp, sig_temp).reshape(time_delayed.shape).T

            # compute the in-phase and quadrature parts
            # NOTE: here we are shifting the in-phase part in time to take into account the delay due to STHT filter
            sig_in_vec_h = np.roll(sig_in_vec, self.kernel_length // 2, axis=0) + 1j * lfilter(self.kernel, [1], sig_in_vec, axis=0)

            # remove the low-pass part of the signal so that STHT works well
            b, a = self.bandpass_filter
            sig_in_vec_h = lfilter(b, a, sig_in_vec_h, axis=0)

            # obtain the real-valued version of the signal
            # dim: `T x 2 num_chan`
            sig_in_real = np.hstack([np.real(sig_in_vec_h), np.imag(sig_in_vec_h)])

            # obtain the spike encoding
            spikes_vec = self.spk_encoder.evolve(sig_in_real)

            # compute the filtered version obtained after low-pass filtering the spikes with neuron syn + mem filter
            vmem_vec = lfilter(neuron_impulse_response, [1], spikes_vec, axis=0)

            # extract the stable part of the signal to get rid of transient section
            stable_part = vmem_vec.shape[0] // 4
            vmem_stable = vmem_vec[stable_part:, :]

            # find the suitable beamforming vectors depending on the polarity of the spikes
            if not self.spk_encoder.bipolar:
                # compute the covariance matrix
                C = 1 / vmem_stable.shape[0] * (vmem_stable.T @ vmem_stable)

                # get rid of the DC level of the signal by finding the special singular vector eliminating the DC value
                bf_vec = self._find_dc_removed_sing_vec(C, rel_prec=0.00000001)

            else:

                # compute the covariance matrix
                C = 1 / vmem_stable.shape[0] * (vmem_stable.T @ vmem_stable)

                # arrange the covariance matrix in complex format so that the resulting beamforming vectors are
                # complex rotation invariant as needed for beamforming applications
                dim_comp = C.shape[0] // 2
                C_comp_diag = (C[:dim_comp, :dim_comp] + C[dim_comp:, dim_comp:]) / 2
                C_comp_off = (C[:dim_comp, dim_comp:] + C[dim_comp:, :dim_comp].T) / 2

                C_comp = C_comp_diag + 1j * C_comp_off

                U, D, _ = np.linalg.svd(C_comp)

                bf_vec = np.concatenate([np.real(U[:, 0]), np.imag(U[:, 0])])

            bf_mat.append(bf_vec)

        # beamforming matrix of dim `2 num_mic x num DoA`. Factor 2 in `2 num_mic` is because of working with the real version
        # of the signal rather than the complex version
        bf_mat = np.asarray(bf_mat).T

        return bf_mat

    def apply_to_template(self, bf_mat: np.ndarray, template: Tuple[np.ndarray, np.ndarray, Union[Number, np.ndarray]],
                          snr_db: float) -> np.ndarray:
        """
        this module applies beamforming when the array receives a given template [time_temp, sig_temp, doa_temp] and returns the signal after beamforming.
        Notes: in this version, the DoA of the source transmitting the template signal may vary during time.

        Args:
            bf_mat (np.ndarray): matrix of dim `2 num_mic x num DoA` containing beamforming vectors at various angles.
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

        # input signal of dim `T x num_mic`
        sig_in_vec = np.interp(time_delayed.ravel(), time_temp, sig_temp).reshape(time_delayed.shape).T

        # add noise to the received signal in the array
        noise = np.sqrt(np.mean(sig_in_vec ** 2)) / np.sqrt(snr) * np.random.randn(*sig_in_vec.shape)
        sig_in_vec += noise

        vmem_vec_beamformed = self.apply_to_signal(bf_mat=bf_mat, sig_in_vec=(time_temp, sig_in_vec))

        return vmem_vec_beamformed

    def apply_to_signal(self, bf_mat: np.ndarray, sig_in_vec: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        this function applies beamforming for a given signal received from array elements.
        Args:
            bf_mat (np.ndarray): beamforming matrix of dimension `2 num_mic x num_DOA`
            sig_in_vec (Tuple[np.ndarray, np.ndarray]): a tuple containing time vector and input signal of dim `T x num_mic`

        Returns:
            signal after being processed by the neurons and beamforming.
        """
        # extract time and signal
        time_vec, sig_in_vec = sig_in_vec

        twice_num_mic, num_grid = bf_mat.shape
        num_mic = twice_num_mic // 2

        T, num_chan = sig_in_vec.shape

        if num_chan != num_mic:
            raise ValueError(
                f"number of channels in the input siganl {num_chan} should be the same as the number of microphones {num_mic}!")

        # check the time and if not sampled properly resample the signal
        if not np.allclose(np.diff(time_vec), 1 / self.fs):
            time_vec_new = np.arange(time_vec[0], time_vec[-1], step=1 / self.fs)

            time_vec_all = np.repeat(time_vec.reshape(1, -1), num_mic, axis=0)
            time_vec_new_all = np.repeat(time_vec_new.reshape(1, -1), num_mic, axis=0)

            sig_in_vec_resampled = np.interp(time_vec_new_all.ravel(), time_vec_all.ravel(),
                                             sig_in_vec.ravel()).reshape(-1, num_mic)

            # replace the original signal
            sig_in_vec = sig_in_vec_resampled
            time_vec = time_vec_new

        # compute the in-phase and quadrature parts
        # NOTE: here we are shifting the in-phase part in time to take into account the delay due to STHT filter
        sig_in_vec_h = np.roll(sig_in_vec, self.kernel_length // 2, axis=0) + 1j * lfilter(self.kernel, [1],
                                                                                           sig_in_vec, axis=0)

        # remove the low-pass part of the signal so that STHT works well
        b, a = self.bandpass_filter
        sig_in_vec_h = lfilter(b, a, sig_in_vec_h, axis=0)

        # obtain the real-valued version of the signal
        # dim: `T x 2 num_mic`
        sig_in_real = np.hstack([np.real(sig_in_vec_h), np.imag(sig_in_vec_h)])

        # obtain the spike encoding
        spikes_vec = self.spk_encoder.evolve(sig_in_real)

        # compute the filtered version after being processed by the neuron
        # build the neuron kernel
        tau_syn, tau_mem = self.tau_vec[0], self.tau_vec[1]
        time_neuron = time_vec - time_vec[0]

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

        # compute the filtered version obtained after low-pass filtering the spikes with neuron syn + mem filter
        vmem_vec = lfilter(neuron_impulse_response, [1], spikes_vec, axis=0)

        # apply beamforming to the final real-valued signal
        # resulting signal of dimension `T x num_grid`
        vmem_vec_beamformed = vmem_vec @ bf_mat

        return vmem_vec_beamformed

    def _find_dc_removed_sing_vec(self, C: np.ndarray, rel_prec: float = 0.0001):
        """This function computes the conditional singular vector of a PSD real-valued matrix C.
        The resulting singular vector should be orthogonal to all-one vector.

        NOTE: this method is needed in computing beamforming vectors for SNNs since spikes and also neurons impulse
        responses are all positive where as a result the signal covariance matrix is unwantedly has a dominant
        singular value along the DC component.

        Args:
            C (np.ndarray): PSD matrix
            rel_prec (float): relative precision  root-finding procedure.

        Returns:
            np.ndarray: the conditional singular vector

        """
        # compute the SVD
        U, D, _ = np.linalg.svd(C)

        all_one = np.ones(C.shape[0])

        theta = U.T @ all_one

        # use root finding to find the position
        u_min = D[1]
        u_max = D[0]

        while True:
            # compute the relative precision
            if (u_max - u_min) / u_min < rel_prec:
                break

            # continue root finding
            u_mid = (u_min + u_max) / 2
            val_mid = np.sum(theta ** 2 / (D - u_mid))

            if val_mid < 0.0:
                u_min = u_mid
            else:
                u_max = u_mid

        # compute the root
        root = (u_min + u_max) / 2.0

        # find the maximum conditioned singular value
        sing_vec = np.einsum('ij, j -> i', U, theta / (D - root))

        # normalize the singular vector
        sing_vec = sing_vec / np.linalg.norm(sing_vec)

        return sing_vec
