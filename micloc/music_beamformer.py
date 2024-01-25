# ----------------------------------------------------------------------------------------------------------------------
# This module implements the conventional MUSIC beamformer.
# It detects the active frequencies in the input signal and applies conventional beamforming to recover the DoA.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 25.01.2024
# ----------------------------------------------------------------------------------------------------------------------
from micloc.array_geometry import ArrayGeometry
from micloc.filterbank import ButterworthFilterbank
import numpy as np

import matplotlib.pyplot as plt

from typing import List, Union, Tuple
from numbers import Number


class MUSIC:
    def __init__(self, geometry: ArrayGeometry, freq_range: np.ndarray, doa_list: np.ndarray,
                 frame_duration: float = 0.25, fs: float = 48_000):
        """
        this module builds a MUSIC beamformer for localization.

        Args:
            geometry (ArrayGeometry): geometry of the array.
            freq_range (np.ndarray): frequency range of the signal processed by the array.
            doa_list (np.ndarray): an array containing a grid of DoAs to be covered during localization.
            frame_duration (float): duration of each frame of signal on which MUSIC is applied. Dafaults to 0.25 sec (250 ms).
            NOTE: during beamforming one can use overlapping frames if needed.
            fs (float): sampling period of the board.
        """

        if len(freq_range) != 2 or freq_range[0] > freq_range[1]:
            raise ValueError("frequency range should be a list containing the minimum and maximum frequency!")
        self.freq_range = np.asarray(freq_range)

        self.doa_list = np.asarray(doa_list)

        self.frame_duration = frame_duration

        self.fs = fs

        # build a filterbank consisting of a single filter for filtering the input signal
        order = 1
        self.filterbank = ButterworthFilterbank(freq_bands=[freq_range], order=order, fs=fs)

        # array geometry
        self.geometry = geometry

    def array_response(self, freq_list: List[float]) -> np.ndarray:
        """this function takes a list of frequencies and computes array response vector at those frequencies.

        Args:
            freq_list (List[float]): a list containing frequencies.

        Returns:
            np.ndarray: array response vector at the provided frequencies at DoAs.
        """

        # compute the physical delays at array elements for a wave coming from given DoAs.
        # NOTE: this is an array of dim `num_mic x num_DoA`.
        delays = np.asarray([self.geometry.delays(theta=theta, normalized=False) for theta in self.doa_list]).T

        # compute the array response vector based on those delays
        # NOTE: this is an array of dim `num_freq x num_mic x num_DoA`
        arr_resp_vec = np.asarray([np.exp(-1j * 2 * np.pi * freq * delays) for freq in freq_list])

        return arr_resp_vec

    def beamforming(self, sig_in: np.ndarray, num_active_freq: int, num_fft_bin: int) -> np.ndarray:
        """this function decomposes the input signal (typically a signal received over a frame) into its narrowband components and applies 
        narrowband beamforming to `num_active_freq` dominant frequencies and accumulates the power to produce angular power spectrum.

        Args:
            sig_in (np.ndarray): input signal of dim `T x num_mic`.
            num_active_freq (int): number of active/dominant frequencies in the spectrum (frequency range) of the signal
            in which beamforming is going to be applied.
            num_fft_bin (int): length of FFT frame in time used for decomposing the signal into its narrowband components.
            NOTE: If `num_fft_bin` is less than signal length, the signal is decomposed into several sub-frames, beamforming is done within each
            sub-frame and then the powers are aggregated to compute the power spectral density.

        Returns:
            np.ndarray: angular power spectrum of dim `num_DoA` illustrating the received power distribution from various DoAs.
        """
        # do some sanity check
        min_freq_spacing = self.fs / num_fft_bin
        fmin, fmax = self.freq_range
        max_num_freq = int((fmax - fmin) / min_freq_spacing)

        if num_active_freq > max_num_freq:
            raise ValueError(
                "number of frequencies is quite large: it may happen that most of these frequencies contain noise!"
            )

        T, num_chan = sig_in.shape
        if num_chan != len(self.geometry):
            raise ValueError("input signal should be of dim `T x num_mic`!")

        # apply filtering to reduce the spectrum of the signal
        # NOTE: since there is a single filter in the filterbank, we choose the first output!
        # the output should be T x num_chan
        sig_in_filt = self.filterbank(sig_in)[0]

        # choose fft bins within signal spectrum
        # we assume that the signal is real-valued and apply conjugate symmetry
        freq_vec = np.linspace(0, self.fs, num_fft_bin)

        # NOTE: since the signal length can be larger/smaller than FFT frame szie we need to:
        # (i) adjust the length
        # (ii) aggregate power over many FFT frames to imporve the estimation precision of power spectral density
        num_fft_frames = T//num_fft_bin

        if num_fft_frames == 0:
            # we zeropad when signal is shorter than FFT length otherwise we truncate the signal
            # to avoid the edge effect of incomplete frames
            num_fft_frames = 1

        sig_fft_len = num_fft_frames * num_fft_bin

        sig_in_filt_adj = np.zeros((sig_fft_len, num_chan))
        sig_in_filt_adj[:sig_fft_len, :] = sig_in_filt[:sig_fft_len, :]

        
        # reshape the signal into frames and apply FFT in each frame
        # NOTE: we first need to apply transpose to transfer the channel indexing to the initial location
        sig_in_fft = np.fft.fft(sig_in_filt_adj.T.reshape(num_chan, num_fft_frames, num_fft_bin), n=num_fft_bin, axis=-1)

        # apply fft and select fft bins within signal spectrum
        fft_bin_index = (fmin <= freq_vec) & (freq_vec <= fmax)
        sig_in_fft_selected = sig_in_fft[:,:,fft_bin_index]
        freq_vec_selected = freq_vec[fft_bin_index]

        # compute the signal energy at the given frequencies and choose dominant ones
        # NOTE: averaging is done across all input channels and FFT frames
        power_in_freq = np.mean(np.abs(sig_in_fft_selected) ** 2, axis=(0,1))
        max_power_indices = np.argsort(power_in_freq)[-num_active_freq:]

        max_power_freq_vec = freq_vec_selected[max_power_indices]
        max_power_sig_in_fft = sig_in_fft_selected[:,:, max_power_indices]

        # compute array response vectors at active frequencies
        # dim: `num_active_freq x num_mic x num_DoA`
        arr_resp = self.array_response(freq_list=max_power_freq_vec)

        # compute angular power spectrum
        ang_pow_spec = 0
        for idx, _ in enumerate(max_power_freq_vec):
            # beamforming is done across first index -> corresponding to input channels in signal (e.g., microphones)
            # since bemaforming matrices are different for different frequencies we need to treat them separately
            # output will be `num_DoA x num_fft_frames` at each frequency
            ang_pow_spec_freq = np.mean(np.abs(np.conj(arr_resp[idx]).T @ max_power_sig_in_fft[:,:,idx]) ** 2, axis=-1)

            # accumulate the power
            ang_pow_spec += ang_pow_spec_freq

        # return the `num_DoA` dim angular power spectrum of the signal across DoAs
        return ang_pow_spec

    def apply_to_signal(self, sig_in: np.ndarray, num_active_freq: int, duration_overlap: float, num_fft_bin: int) -> np.ndarray:
        """this function applies beamforming to overlapping signal frames and returns the estimated angular power spectrum.
        NOTE: since active frequencies in each frame may change, the estimated power spectrum may have jumps from one frame to 
        the next depending on how signal power changes.

        Args:
            sig_in (np.ndarray): input signal.
            num_active_freq (int): number of active/dominant frequencies in signal spectrum on which the beamforming is going to be applied in each frame.
            duration_overlap (float): how much overlap in seconds between consecutive signal frames is allowed.
            num_fft_bin (int): length of FFT frame in time used for decomposing the signal into its narrowband components.

        Returns:
            np.ndarray: angular power spectrum computed in time over each frame.
        """
        # some sanity check
        T, num_chan = sig_in.shape
        num_mic = len(self.geometry)

        if num_chan != num_mic:
            raise ValueError("number of channels in the input signal should be the same as the number of microphones!")

        num_samples_frame = int(self.fs * self.frame_duration)
        num_sample_overlap = int(self.fs * duration_overlap)

        if num_sample_overlap >= num_samples_frame:
            raise ValueError("duration of overlap window is larger than the duration of a single frame!")

        # number of fresh samples in each sliding window
        num_samples_fresh = num_samples_frame - num_sample_overlap

        angular_power_spectrum = []

        idx_slice = 0
        while idx_slice * num_samples_fresh + num_samples_frame <= T:
            start_idx = idx_slice * num_samples_fresh
            end_idx = idx_slice * num_samples_fresh + num_samples_frame

            sig_in_slice = sig_in[start_idx:end_idx, :]
            ang_pow_spec_slice = self.beamforming(sig_in=sig_in_slice, num_active_freq=num_active_freq, num_fft_bin=num_fft_bin)

            angular_power_spectrum.append(ang_pow_spec_slice)

            idx_slice += 1
        
        # check if there are any leftover signal and if it is long enough
        start_idx = idx_slice * num_samples_fresh

        if (T-start_idx)>0.5*num_samples_frame:
            sig_in_slice = sig_in[start_idx:T, :]
            ang_pow_spec_slice = self.beamforming(sig_in=sig_in_slice, num_active_freq=num_active_freq, num_fft_bin=num_fft_bin)

            angular_power_spectrum.append(ang_pow_spec_slice)

        angular_power_spectrum = np.asarray(angular_power_spectrum)

        return angular_power_spectrum

    def apply_to_template(self, template: Tuple[np.ndarray, np.ndarray, Union[Number, np.ndarray]],
                          num_active_freq: int, duration_overlap: float, num_fft_bin: int,
                          snr_db: float) -> np.ndarray:
        """
        this module applies beamforming when the array receives a given template [time_temp, sig_temp, doa_temp] and returns the estimated angular power spectrum.

        Args:
            template Tuple[np.ndarray, np.ndarray, Union[Number, np.ndarray]]): input template signal consisting of time-of-arrival, signal samples and direction of arrival.
            num_active_freq (int): number of dominant/active frequencies over which the beamforming should be carried out.
            duration_overlap (float): how much overlap (in sec) between consecutive frames is allowed.
            num_fft_bin (int): length of FFT frame in time used for decomposing the signal into its narrowband components.
            snr_db (float): signal-to-noise ratio at each array element (microphone).

        Returns:
            np.ndarray: estimated angular power spectrum in time.
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

        # compute the signal received at the array NOTE: here we apply a delay normalization to all the samples later
        # rather than sample by sample normalization which will yield wrong results.
        delays = np.asarray([self.geometry.delays(theta=doa, normalized=False) for doa in doa_temp])
        delays = delays - delays.min()

        time_delayed = time_temp.reshape(-1, 1) - delays
        time_delayed[time_delayed < time_temp.min()] = time_temp.min()

        sig_in_vec = np.interp(time_delayed[:], time_temp, sig_temp).reshape(time_delayed.shape)

        # add noise to the received signal in the array
        noise = np.sqrt(np.mean(sig_in_vec ** 2)) / np.sqrt(snr) * np.random.randn(*sig_in_vec.shape)
        sig_in_vec += noise

        # estimate angular power spectrum in time
        ang_pow_spec = self.apply_to_signal(sig_in=sig_in_vec, num_active_freq=num_active_freq,
                                            duration_overlap=duration_overlap, num_fft_bin=num_fft_bin)

        return ang_pow_spec

