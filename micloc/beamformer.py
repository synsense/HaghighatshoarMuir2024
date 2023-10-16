# ----------------------------------------------------------------------------------------------------------------------
# This module allows to design beamforming matrices for various array geometries.
#
# Note: this module uses the conventional subspace and super-resolution methods for beamforming.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 12.10.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from micloc.array_geometry import ArrayGeometry
from typing import Tuple, Union
from scipy.signal import hilbert, lfilter, butter
from scipy.linalg import eigh
from typing import List
from numbers import Number
from tqdm import tqdm


class Beamformer:
    def __init__(self, geometry: ArrayGeometry, kernel_duration: float, freq_range: List, fs: float):
        """
        this class builds methods for building beamforming matrices in multi-mic arrays based on conventional subspace
        and super-resolution methods.
        Args:
            geometry (ArrayGeometry): encoding of the geometry of the array in terms of time-of-arrivals.
            kernel_duration (float): length of the Hilbert kernel used for localization.
            freq_range (List): a list containing the lower and higher frequency range of the beamformer.
            fs (float): sampling rate of the array.
        """
        self.geometry = geometry
        self.kernel_duration = kernel_duration
        self.fs = fs

        # design the Hilbert kernel
        ker_len = int(fs * kernel_duration)
        impulse = np.zeros(ker_len)
        impulse[0] = 1

        self.kernel = np.fft.fftshift(np.imag(hilbert(impulse)))

        self.freq_range = np.asarray(freq_range)

        # build a bandpass filter to get rid of lower part of frequency causing issues in STHT
        f_low, f_high = freq_range
        try:
            f_low, f_high = freq_range
            if f_low > f_high:
                raise Exception()
        except Exception:
            raise ValueError("freq_range should be a vector consisting of two frequencies f_low < f_high!")

        order = 2
        cutoff = freq_range
        self.bandpass_filter = butter(order, cutoff, btype="bandpass", analog=False, output="ba", fs=fs)

    def design_from_template(self, template: Tuple[np.ndarray, np.ndarray], doa_list: np.ndarray,
                             interference_removal: bool = False) -> np.ndarray:
        """
        this function builds suitable beamforming matrices for the array when it receives a given waveform.
        Args:
            template (Tuple[np.ndarray, np.ndarray]): a tuple containing the template signal (time_temp, sig_temp).
            doa_list (np.ndarray): a list containing target DoAs for which the beamforming is designed.
            interference_removal (bool): this is a flag showing if the interference of neighboring angles should be taken
            into account in the design of beamforming vectors.

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

        # list of covariance matrices
        cov_mat_list = []

        print()
        print('+' * 150)
        print(" designing beamforming matrices for various DoAs ".center(150, '+'))
        print('+' * 150)

        for doa in tqdm(doa_list):
            # compute the delays associated with the DoA
            delays = self.geometry.delays(
                theta=doa,
                normalized=True
            )

            # interpolate the input signal with the given delay values
            sig_in_vec = []

            for delay in delays:
                time_delay = time_temp - delay
                time_delay[time_delay < time_temp.min()] = time_temp.min()

                sig_in = np.interp(time_delay, time_temp, sig_temp)

                sig_in_vec.append(sig_in)

            # format: T x num_chan (microphone)
            sig_in_vec = np.asarray(sig_in_vec).T

            # compute STHT
            sig_in_vec_h = np.roll(sig_in_vec, len(self.kernel) // 2, axis=0) + 1j * lfilter(self.kernel, [1],
                                                                                             sig_in_vec, axis=0)

            # get rid of the low and high part of the spectrum
            b, a = self.bandpass_filter
            sig_in_vec = lfilter(b, a, sig_in_vec, axis=0)

            # now that the input signals in all arrays are available, design the beamformer
            # 1. remove the transient part
            stable_part = min([len(self.kernel), sig_in_vec_h.shape[0] // 2])
            sig_in_vec_h_stable = sig_in_vec_h[stable_part:, :]

            # 2. compute the covariance matrix
            cov_mat = 1 / sig_in_vec_h_stable.shape[0] * (sig_in_vec_h_stable.conj().T @ sig_in_vec_h_stable)

            cov_mat_list.append(cov_mat)

        # now we have access to all covariance matrices: design beamforming vectors
        bf_mat = []

        if not interference_removal:
            # no interference removal is needed: just apply ordinary SVD to extract the beamforming vectors
            for cov_mat in cov_mat_list:
                U, D, _ = np.linalg.svd(cov_mat)
                bf_mat.append(U[:, 0])

            bf_mat = np.asarray(bf_mat).T

        else:
            # get rid of interference due to other DoAs

            # compute the sum of all covariance matrices
            cov_mat_sum = 0
            for cov_mat in cov_mat_list:
                cov_mat_sum = cov_mat_sum + cov_mat

            # add a little bit diagonal offset to make sure that the matrix is strictly PSD
            cov_mat_sum += np.diag(np.mean(np.diag(cov_mat_sum)) * np.ones(cov_mat_sum.shape[0])) / 10

            for idx, cov_mat in enumerate(cov_mat_list):
                # in scipy the singular values are sorted in an ascending fashion and not the same as those in SVD
                # apply generalized eigen vector problem
                D, U = eigh(cov_mat, cov_mat_sum - cov_mat)

                # choose the last vector
                bf_vec = U[:, -1]
                bf_vec = bf_vec / np.linalg.norm(bf_vec)

                bf_mat.append(bf_vec)

            bf_mat = np.asarray(bf_mat).T

        return bf_mat, cov_mat_list

    def apply_to_template(self, bf_mat: np.ndarray, template: Tuple[np.ndarray, np.ndarray, Union[Number, np.ndarray]],
                          snr_db: float) -> np.ndarray:
        """
        this module applies beamforming when the array receives a given template [time_temp, sig_temp, doa_temp] and returns the signal after beamforming.
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
        # NOTE: here we apply a delay normalization to all the samples later rather than sample by sample normalization which will yield wrong results.
        delays = np.asarray([self.geometry.delays(theta=doa, normalized=False) for doa in doa_temp]).T
        delays = delays - delays.min()

        time_delayed = time_temp.reshape(-1, 1) - delays
        time_delayed[time_delayed < time_temp.min()] = time_temp.min()

        sig_in_vec = np.interp(time_delayed[:], time_temp, sig_temp).reshape(time_delayed.shape)

        # add noise to the received signal in the array
        noise = np.sqrt(np.mean(sig_in_vec ** 2)) / np.sqrt(snr) * np.random.randn(*sig_in_vec.shape)
        sig_in_vec += noise

        # compute signal after beamforming
        sig_in_beamformed = self.apply_to_signal(bf_mat=bf_mat, sig_in=sig_in_vec)

        return sig_in_beamformed

    def apply_to_signal(self, bf_mat: np.ndarray, sig_in: np.ndarray) -> np.ndarray:
        """
        this function applies bemforming for a given signal received from array elements.
        Args:
            bf_mat (np.ndarray): beamforming matrix of dim `num_mic x num_grid`.
            sig_in (np.ndarray): input data of dim `T x num_mic`

        Returns:
            signal after beamforming of dim `T x num_grid`.
        """

        num_mic, num_grid = bf_mat.shape

        T, num_chan = sig_in.shape

        if num_chan != num_mic:
            raise ValueError(
                f"number of channels in the input siganl {num_chan} should be the same as the number of microphones {num_mic}!")

        # compute the kernel Hilbert transform of the input signal
        sig_in_kernel_H = np.roll(sig_in, len(self.kernel) // 2, axis=0) + 1j * lfilter(self.kernel, [1], sig_in,
                                                                                        axis=0)

        # apply low-pass filteirng to get rid of low and high frequency part of the spectrum
        b, a = self.bandpass_filter
        sig_in_kernel_H = lfilter(b, a, sig_in_kernel_H, axis=0)

        # apply beamforming to obtain beamformed signal of dim `T x G`
        sig_bf = sig_in_kernel_H @ bf_mat.conj()

        return sig_bf
