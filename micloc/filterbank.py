# ----------------------------------------------------------------------------------------------------------------------
# This module simulates a simple filterbank module for processing multi-mic audio signals.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 12.10.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from typing import List
from scipy.signal import lfilter, butter


class Filterbank:
    def __init__(self, ba_list: List):
        """
        This module builds a filterbank for processing multi-mic data.
        Args:
            ba_list (List): a list containing (b,a) parameters of the filters.
        """
        self.ba_list = ba_list

    def evolve(self, sig_in: np.ndarray):
        """
        this function applies the filterbank to an T x M signal and produces an F x T x M signal containing F channels in M microphones.
        Args:
            sig_in (np.ndarray): T x M signal received from M microphones.

        Returns:
            np.ndarray: an array of dimension F x T x M containing the signal in F frequency channels, M microphones across T time steps.
        """
        # check single channel signals
        if len(sig_in.shape) == 1:
            sig_in = sig_in.reshape(-1, 1)

        sig_out = []

        for b, a in self.ba_list:
            sig_out_freq_channel = lfilter(b, a, sig_in, axis=0)
            sig_out.append(sig_out_freq_channel)

        sig_out = np.asarray(sig_out)

        return sig_out

    def __call__(self, *args, **kwargs):
        """this is the same as evolve function."""
        return self.evolve(*args, **kwargs)

    def __len__(self):
        """returns the number of filters in the filterbank."""
        return len(self.ba_list)


class ButterworthFilterbank(Filterbank):
    def __init__(self, freq_bands: List, order: int, fs: float):
        """
        This modules builds a Butterworth filterbank of given order covering various frequency bands.

        Args:
            freq_bands (List): a list containing frequency bands covered by the filterbank. Each frequency band consists of
            two frequencies [f_min, f_max] showing the cutoff or 3dB frequencies of the filter.
            order (int): order of the filters in the filterbank.
            fs (float): sampling frequency of the filter.
        """
        self.order = order
        self.fs = fs

        self.freq_bands = np.asarray(freq_bands)
        if len(self.freq_bands.shape) == 1:
            # only a single band is provided
            self.freq_bands = self.freq_bands.reshape(1, -1)

        # build the filters
        ba_list = []

        for freq_band in freq_bands:
            b, a = butter(order, freq_band, btype="bandpass", output="ba", fs=fs)
            ba_list.append((b, a))

        # build the core filterbank with the given ba_list
        super().__init__(ba_list=ba_list)
