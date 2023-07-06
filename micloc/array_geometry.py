# ----------------------------------------------------------------------------------------------------------------------
# This module implements a localization method for multi-mic data based on Hilbert transform.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.07.2023
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.signal import lfilter, butter


class ArrayGeometry:
    def __init__(self, r_vec: np.ndarray, theta_vec: np.ndarray, speed: float = 340):
        """
        class for encoding the array geometry in terms of delays at various DoAs.
        Args:
            r_vec (np.ndarray): array containing the distances of the array elements in polar coordinate.
            theta_vec (np.ndarray): array containing the angles of the elements in polar coordinate.
            speed (float, optional): speed of wave captured by the array. Defaults to 340 m/s.
        """
        if np.any(r_vec < 0):
            raise ValueError("distances of the elements in `r_vec` should be all positive!")
        self.r_vec = r_vec
        self.theta_vec = theta_vec

        self.speed = speed

    def delays(self, theta: float, normalized: bool = True) -> np.ndarray:
        """
        this function returns the relative delay of array elements for a wave with DoA `angle`.
        Args:
            theta (float): DoA of the incoming wave.
            normalized (bool, optional): a flag showing if the delays are normalized to start from 0.

        Returns:
            an array containing the delays of the array elements
        """
        delays = - self.r_vec * np.cos(self.theta_vec - theta) / self.speed

        if normalized:
            delays -= np.min(delays)

        return delays


class CircularArray(ArrayGeometry):
    def __init__(self, radius: float, num_mic: int, speed: float = 340):
        """
        class encoding the geometry of a circular array.
        Args:
            radius (float): radius of the array.
            num_mic (int): number of microphones in the array.
            speed (float, optional): speed of wave captured by the array. Defaults to 340 m/s.
        """
        r_vec = radius * np.ones(num_mic)
        theta_vec = np.linspace(0, 2 * np.pi, num_mic)

        super().__init__(r_vec=r_vec, theta_vec=theta_vec, speed=speed)


class LinearArray(ArrayGeometry):
    def __init__(self, spacing: float, num_mic: int, speed: float = 340):
        """
        class for encoding the array geometry for a linear array.
        Args:
            spacing (float): the spacing between array elements.
            num_mic (int): number of microphones in the array.
        """
        r_vec = spacing * np.arange(num_mic)
        theta_vec = np.pi / 2 * np.ones(num_mic)

        super().__init__(r_vec=r_vec, theta_vec=theta_vec, speed=speed)
