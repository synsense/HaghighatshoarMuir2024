# ----------------------------------------------------------------------------------------------------------------------
# This module allows to model the effect of propagation in the environment using array geometry.
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 12.10.2023
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.signal import lfilter, butter

SOUND_SPEED_IN_OPEN_AIR = 340


class ArrayGeometry:
    def __init__(
        self,
        r_vec: np.ndarray,
        theta_vec: np.ndarray,
        speed: float = SOUND_SPEED_IN_OPEN_AIR,
    ):
        """
        class for encoding the array geometry in terms of delays at various DoAs.
        Args:
            r_vec (np.ndarray): array containing the distances of the array elements in polar coordinate.
            theta_vec (np.ndarray): array containing the angles of the elements in polar coordinate.
            speed (float, optional): speed of wave captured by the array. Defaults to 340 m/s.
        """
        if np.any(r_vec < 0):
            raise ValueError(
                "distances of the elements in `r_vec` should be all positive!"
            )
        self.r_vec = r_vec
        self.theta_vec = theta_vec

        self.speed = speed

    def delays(self, theta: float, normalized: bool = True) -> np.ndarray:
        """
        this function returns the relative delay of array elements for a wave with DoA `theta`.
        Args:
            theta (float): DoA of the incoming wave.
            normalized (bool, optional): a flag showing if the delays are normalized to start from 0.
            NOTE: this may be good to avoid shifting all signal samples. But should be avoided if some comparison between
            various DoAs is made since in that case this normalization creates issues.

        Returns:
            an array containing the delays of the array elements
        """
        delays = -self.r_vec * np.cos(self.theta_vec - theta) / self.speed

        if normalized:
            delays -= np.min(delays)

        return delays

    def __len__(self) -> int:
        """ " returns the number of sensors in the array"""
        return len(self.r_vec)


class CircularArray(ArrayGeometry):
    def __init__(
        self, radius: float, num_mic: int, speed: float = SOUND_SPEED_IN_OPEN_AIR
    ):
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


class CenterCircularArray(ArrayGeometry):
    def __init__(
        self, radius: float, num_mic: int, speed: float = SOUND_SPEED_IN_OPEN_AIR
    ):
        """
        class encoding the geometry of a circular array which has one of microphones put at the center.
        Args:
            radius (float): radius of the array.
            num_mic (int): number of microphones.
            speed (float): speed of wave captured by the array. Defaults to 340 m/s.
        """
        r_vec = np.array([*list(radius * np.ones(num_mic - 1)), 0.0])
        theta_vec = np.array([*list(np.linspace(0, 2 * np.pi, num_mic - 1)), 0.0])
        super().__init__(r_vec=r_vec, theta_vec=theta_vec, speed=speed)


class LinearArray(ArrayGeometry):
    def __init__(
        self, spacing: float, num_mic: int, speed: float = SOUND_SPEED_IN_OPEN_AIR
    ):
        """
        class for encoding the array geometry for a linear array.
        Args:
            spacing (float): the spacing between array elements.
            num_mic (int): number of microphones in the array.
        """
        r_vec = spacing * np.arange(num_mic)
        theta_vec = np.pi / 2 * np.ones(num_mic)

        super().__init__(r_vec=r_vec, theta_vec=theta_vec, speed=speed)


class Random2DArray(ArrayGeometry):
    def __init__(
        self, radius: float, num_mic: int, speed: float = SOUND_SPEED_IN_OPEN_AIR
    ):
        r_vec = np.sqrt(np.random.rand(num_mic)) * radius
        theta_vec = np.random.rand(num_mic) * 2 * np.pi

        super().__init__(r_vec=r_vec, theta_vec=theta_vec, speed=speed)

        self.radius = radius
