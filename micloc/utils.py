# ----------------------------------------------------------------------------------------------------------------------
# This module implements several utility functions for beamforming.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.07.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import warnings


class Envelope:
    def __init__(self, rise_time: float, fall_time: float, fs: float):
        """
        this module implements a simple envelope extractor for signal processing applications.
        Args:
            rise_time (float): time-constant of the low-pass filter during the rise period.
            fall_time (float): time-constant of the low-pass filter during the fall period.
            fs (float): sampling rate of the module.
        """
        if rise_time > fall_time:
            raise ValueError("for proper functioning, an envelope estimator should have a larger fall time!")

        self.rise_time = rise_time
        self.fall_time = fall_time
        self.fs = fs

        # parameters of the low-pass filter in rise and fall mode
        self.win_lens = np.asarray([int(fs * fall_time), int(fs * rise_time)])

    def evolve(self, sig_in: np.ndarray) -> np.ndarray:
        T, channel = sig_in.shape

        if T < channel:
            warnings.warn("number of channels in the input signal is larger than number of samples in each channel!")

        # compute the abs of the signal for envelope estimation
        sig_in = np.abs(sig_in)

        state = np.copy(sig_in[0])

        sig_out = []

        for sig in sig_in[1:]:
            sig_out.append(state[:])

            # rise or fall
            rise_or_fall = (sig >= state).astype(int)

            # update the state depending on rise or fall
            win_len_state = self.win_lens[rise_or_fall]

            # signal should not be fed in rise mode
            state = (1 - 1 / win_len_state) * state + 1 / win_len_state * sig * rise_or_fall

        # append the last state
        sig_out.append(state[:])

        sig_out = np.asarray(sig_out)

        return sig_out
