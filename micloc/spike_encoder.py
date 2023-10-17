# ----------------------------------------------------------------------------------------------------------------------
# This module builds various spike encoders that can be useful for array processing applications.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 12.10.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.signal import find_peaks


class SpikeEncoder:
    def __init__(self):
        pass

    def evolve(self, sig_in: np.ndarray) -> np.ndarray:
        raise NotImplementedError("this methods needs to be implemented in various spike encoders!")


class IAFSpikeEncoder(SpikeEncoder):
    def __init__(self, target_spike_rate: float, fs: float):
        """
        this class builds an IAF spike encoder for multi-mic signals.
        NOTE: this type of spike encoder seems to be really good for speech applications but lacks sufficient precision
        in localization applications.

        Args:
            target_rate (float): target rate of spikes.
            fs (float): sampling clock rate of the array.
        """
        self.target_spike_rate = target_spike_rate
        self.fs = fs

    def evolve(self, sig_in: np.ndarray) -> np.ndarray:
        """
        this function converts the input signal into spikes.
        Args:
            sig_in (np.ndarray): T x num_chan signal containing the signal received from `num_chan` channels, e.g., microphones.

        Returns:
            np.ndarray: array containing the corresponding spike encoding.
        """
        # compute the required threshold
        threshold = np.mean(np.abs(sig_in)) * self.fs / self.target_spike_rate

        # compute cumsum and spike location
        sum_power = np.cumsum(np.abs(sig_in), axis=0)

        spikes = np.diff(np.floor(sum_power / threshold), axis=0)

        return spikes


class IAFZeroCrossingSpikeEncoder:
    def __init__(self, target_spike_rate: float, fs: float):
        """
        this class builds an IAF spike encoder for multi-mic signals.

        The difference with the conventional IAF spike encoder
        is that IAF spike encoding is applied to the cumsum of the input signal so that the high-freq part of the spikes
        corresponds to maximum amplitude of the signal at which the signal to noise ratio is quite high.

        Args:
            target_rate (float): target rate of spikes.
            fs (float): sampling clock rate of the array.
        """
        self.target_spike_rate = target_spike_rate
        self.fs = fs

    def evolve(self, sig_in: np.ndarray) -> np.ndarray:
        """
        this function converts the input signal into spikes.
        Args:
            sig_in (np.ndarray): T x num_chan signal containing the signal received from num_chan channels, e.g., microphones.

        Returns:
            np.ndarray: corresponding spike encoding.
        """
        # compute the required threshold
        sig_in_cs = np.cumsum(sig_in, axis=0)
        threshold = np.mean(np.abs(sig_in_cs)) * self.fs / self.target_spike_rate

        # compute cumsum and spike location
        sum_power = np.cumsum(np.abs(sig_in_cs), axis=0)

        spikes = np.diff(np.floor(sum_power / threshold), axis=0)

        return spikes


class ZeroCrossingSpikeEncoder(SpikeEncoder):
    def __init__(self, fs: float, robust_width: int = 1, bipolar: bool = False):
        """
        this class builds a robust spike encoder for multi-mic signals.
        The Zerocrossing can be unipolar or bipolar depending on the application.

        Args:
            fs (float): sampling rate of the input signal.
            robust_width (int): length of the window over which the robust zerocrossing is estimated.
            bipolar (bool): is zerocrossing should targets the valleys as well as peaks of the signal. Defaults to False.
        """
        self.fs = fs
        self.robust_width = robust_width
        self.bipolar = bipolar

    def evolve(self, sig_in: np.ndarray) -> np.ndarray:
        """
        this function converts the input signal into spikes.
        Args:
            sig_in (np.ndarray): T x num_chan signal containing the signal received from num_chan channels, e.g., microphones.

        Returns:
            np.ndarray: corresponding spike encoding.
        """
        spikes = np.zeros_like(sig_in).T

        for chan, sig_chan in enumerate(sig_in.T):
            peaks, _ = find_peaks(np.cumsum(sig_chan), distance=self.robust_width)
            spikes[chan, peaks] = 1

            # in the bipolar version: we also encode the negative spikes
            if self.bipolar:
                valleys, _ = find_peaks(-np.cumsum(sig_chan), distance=self.robust_width)
                spikes[chan, valleys] = -1

        return spikes.T


class PeakSpikeEncoder(SpikeEncoder):
    def __init__(self, fs: float):
        """
        this class builds a robust spike encoder for multi-mic signals.
        Args:
            fs (float): sampling rate of the input signal.
        """
        self.fs = fs

    def evolve(self, sig_in: np.ndarray, robust_width: int = 1) -> np.ndarray:
        """
        this function converts the input signal into spikes.
        Note: compared with zerocrossing that uses the peaks of the cumsum of the signal, this method uses the peak locations
        directly for spike encoding.
        Args:
            sig_in (np.ndarray): T x num_chan signal containing the signal received from num_chan microphones.
            robust_width (int): length of the window over which the robust zerocrossing is estimated.

        Returns:
            np.ndarray: corresponding spike encoding.
        """
        spikes = np.zeros_like(sig_in).T

        for chan, sig_chan in enumerate(sig_in.T):
            peaks, _ = find_peaks(sig_chan, distance=robust_width)
            spikes[chan, peaks] = 1

        return spikes.T
