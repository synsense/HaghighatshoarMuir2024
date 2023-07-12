# ----------------------------------------------------------------------------------------------------------------------
# This module builds an IAF spike encoder for array processing applications.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 10.07.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np


class IAFSpikeEncoder:
    def __init__(self, target_spike_rate: float, fs: float):
        """
        this class builds an IAF spike encoder for multi-mic signals.
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
            sig_in (np.ndarray): T x num_chan signal containing the signal received from num_chan microphones.

        Returns:
            np.ndarray: corresponding spike encoding.
        """
        # compute the required threshold
        threshold = np.mean(np.abs(sig_in)) * self.fs / self.target_spike_rate

        # compute cumsum and spike location
        sum_power = np.cumsum(np.abs(sig_in), axis=0)

        spikes = np.diff(np.floor(sum_power/threshold), axis=0)

        return spikes
