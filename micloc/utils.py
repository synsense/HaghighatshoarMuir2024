# ----------------------------------------------------------------------------------------------------------------------
# This module implements several utility functions used in beamforming and localization.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 16.10.2023
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
            raise ValueError(
                "for proper functioning, an envelope estimator should have a larger fall time!"
            )

        self.rise_time = rise_time
        self.fall_time = fall_time
        self.fs = fs

        # parameters of the low-pass filter in rise and fall mode
        self.win_lens = np.asarray([int(fs * fall_time), int(fs * rise_time)])

    def evolve(self, sig_in: np.ndarray) -> np.ndarray:
        """
        this module computes and tracks the envelope of the signal across various channels which allows to do
        a robust estimation of active channels.

        Args:
            sig_in (np.ndarray): input signal of dimension `T x num_chan`.

        Returns:
            np.ndarray: array containing extracted envelopes in channels.
        """

        T, channel = sig_in.shape

        if T < channel:
            warnings.warn(
                "number of channels in the input signal is larger than number of samples in each channel!"
            )

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
            state = (
                1 - 1 / win_len_state
            ) * state + 1 / win_len_state * sig * rise_or_fall

        # append the last state
        sig_out.append(state[:])

        sig_out = np.asarray(sig_out)

        return sig_out


def find_peak_location(sig_in: np.ndarray, win_size: int, periodic: bool = True) -> int:
    """
    this function finds the location of peak in the input signal by applying some averaging.
    Args:
        sig_in (np.ndarray): input signal.
        win_size (int): size of averaging window used for finding the location of peak.
        periodic (bool): if the signal is periodic or not!

    Returns:
        int: location of peak.
    """

    if sig_in.ndim != 1:
        raise ValueError("input signal should be 1-dim!")

    if win_size % 2 != 1:
        raise ValueError(
            "averaging window size should be odd to not create confusion in peak index!"
        )

    if win_size > len(sig_in) // 2:
        raise ValueError(
            "size of averaging window is larger than half the length of input signal!"
        )

    # averaging filter
    window = np.ones(win_size)

    sig_avg = np.convolve(window, sig_in, mode="full")

    # find the location of peak
    index = np.argmax(sig_avg)
    index -= win_size // 2

    if periodic:
        index = index % len(sig_in)

    return index
