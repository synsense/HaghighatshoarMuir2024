# ----------------------------------------------------------------------------------------------------------------------
# This module allows to design beamforming matrices for various array geometries.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.07.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from micloc.array_geometry import ArrayGeometry
from typing import Tuple, Union
from scipy.signal import hilbert, lfilter
from numbers import Number
from tqdm import tqdm


class Beamformer:
    def __init__(self, geometry: ArrayGeometry, kernel_duration: float, fs: float):
        """
        this class builds methods for building beamforming matrices in multi-mic arrays.
        Args:
            geometry (ArrayGeometry): encoding of the geometry of the array in terms of time-of-arrivals.
            kernel_duration (float): length of the Hilbert kernel used for localization.
            fs (float): sampling rate of the array.
        """
        self.geometry = geometry
        self.kernel_duration = kernel_duration
        self.fs = fs

        # design the Hilbert kernel
        ker_len = int(fs * kernel_duration)
        impulse = np.zeros(ker_len)
        impulse[0] = 1

        self.kernel = np.imag(hilbert(impulse))

    def design_from_template(self, template: Tuple[np.ndarray, np.ndarray], doa_list: np.ndarray) -> np.ndarray:
        """
        this function builds suitable beamforming matrices for the array when it receives a given waveform.
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

        # matrix containing dominant directions in its colums
        bf_mat = []

        print()
        print('+'*150)
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

            sig_in_vec = np.asarray(sig_in_vec)

            # compute the Hilbert transform
            sig_in_vec_h = sig_in_vec + 1j * lfilter(self.kernel, [1], sig_in_vec, axis=1)

            # now that the input signals in all arrays are available, design the beamformer
            # 1. remove the transient part
            stable_part = min([len(self.kernel), sig_in_vec_h.shape[1]//2])
            sig_in_vec_h_stable = sig_in_vec_h[:, stable_part:]

            # 2. compute the covariance matrix
            cov_mat = 1 / sig_in_vec_h_stable.shape[1] * (sig_in_vec_h_stable @ sig_in_vec_h_stable.conj().T)

            U, D, _ = np.linalg.svd(cov_mat)

            bf_mat.append(U[:, 0])

        bf_mat = np.asarray(bf_mat).T

        return bf_mat

    def apply_to_template(self, bf_mat: np.ndarray, template: Tuple[np.ndarray, np.ndarray, Union[Number, np.ndarray]], snr_db:float) -> np.ndarray:
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
            raise ValueError("input template should be a tuple containing (time_in, sig_in, doa_in) of the template signal!")

        if isinstance(doa_temp, Number):
            doa_temp = doa_temp * np.ones_like(sig_temp)

        # compute SNR
        snr = 10**(snr_db/10)

        # resample the signal to the clock rate of the array
        time_in = np.arange(time_temp.min(), time_temp.max(), step=1 / self.fs)
        sig_in = np.interp(time_in, time_temp, sig_temp)
        doa_in = np.interp(time_in, time_temp, doa_temp)

        time_temp, sig_temp, doa_temp = time_in, sig_in, doa_in

        # compute the signal received at the array
        # NOTE: here we apply a delay normalization to all the samples later rather than sample by sample normalization which will yield wrong results.
        delays = np.asarray([self.geometry.delays(theta=doa, normalized=False) for doa in doa_temp]).T
        delays = delays - delays.min()

        time_delayed = time_temp.reshape(1, -1) - delays
        time_delayed[time_delayed < time_temp.min()] = time_temp.min()

        sig_in_vec = np.interp(time_delayed[:], time_temp, sig_temp).reshape(time_delayed.shape)

        # add noise to the received signal in the array
        noise = np.sqrt(np.mean(sig_in_vec**2))/np.sqrt(snr) * np.random.randn(*sig_in_vec.shape)
        sig_in_vec += noise

        # compute the Hilbert transform
        sig_in_vec_h = sig_in_vec + 1j * lfilter(self.kernel, [1], sig_in_vec, axis=1)

        # apply the beamforming
        sig_in_beamformed = bf_mat.conj().T @ sig_in_vec_h

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
            raise ValueError(f"number of channels in the input siganl {num_chan} should be the same as the number of microphones {num_mic}!")

        # compute the kernel Hilbert transform of the input signal
        data_in_kernel_H = data_in + 1j * lfilter(self.kernel, [1], data_in, axis=1)

        # apply bemaforming
        data_bf = bf_mat.conj().T @ data_in_kernel_H

        return data_bf

