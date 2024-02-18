# ----------------------------------------------------------------------------------------------------------------------
# This module implements a live demo based on MUSIC method.
# It detects the active frequencies in the input signal and applies conventional beamforming to recover the DoA.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 25.01.2024
# ----------------------------------------------------------------------------------------------------------------------
from micloc.record import AudioRecorder
from micloc.visualizer import Visualizer

from micloc.array_geometry import ArrayGeometry, CenterCircularArray
from micloc.music_beamformer import MUSIC
import numpy as np


class Demo:
    def __init__(
        self,
        geometry: ArrayGeometry,
        freq_range: np.ndarray,
        num_active_freq: int,
        doa_list: np.ndarray,
        recording_duration: float = 0.25,
        num_fft_bin: int = 2048,
        fs: float = 48_000,
    ):
        """
        this module builds a MUSIC beamformer for localization.

        Args:
            geometry (ArrayGeometry): geometry of the array.
            freq_range (np.ndarray): frequency range of the signal processed by the array.
            num_active_freq (int): number of active frequencies used for localization in the demo.
            doa_list (np.ndarray): an array containing a grid of DoAs to be covered during localization.
            recording_duration (float): duration of each frame of recording of signal on which MUSIC is applied. Deafults to 0.25 sec (250 ms).
            NOTE: during beamforming one can use overlapping frames if needed.
            num_fft_bin (int): number of FFT bins used in MUSIC. Defaults to 2048.
            fs (float): sampling period of the board.
        """

        # build the music module
        self.music = MUSIC(
            geometry=geometry,
            freq_range=freq_range,
            doa_list=doa_list,
            frame_duration=recording_duration,
            fs=fs,
        )

        # recording params
        self.doa_list = doa_list
        self.num_active_freq = num_active_freq
        self.recording_duration = recording_duration
        self.num_fft_bin = num_fft_bin
        self.fs = fs

    def estimate_doa(self, angular_power_spec: np.ndarray, method: str) -> float:
        """this method allows to estimate the DoA from the angular power spectrum.
        NOTE: each method may be good for one scenario but may not be good when there are several targets.
        This should be known a priori.

        Args:
            angular_power_spec (np.ndarray): angular power spectrum of the input signal estimated through beamforming.
            method (str): name of the method. Options are:
                - "peak" : use the DoA corresponding to the peak value of spike rate.
                - "periodic_ml" : this method treats DoA as a periodic function and applies ML estimate to it.
                - "trimmed_periodic_ml" : the method chooses the spike rates around the peak value and estimates DoA
                   using periodic ML method.
        Returns:
            float: estimated DoA.
        """
        # possible methods
        method_list = ["peak", "periodic_ml", "trimmed_periodic_ml"]
        if method not in method_list:
            raise ValueError(
                f"only the following estimation methods are supported:\n{method_list}"
            )

        if method == "peak":
            DoA_index = np.argmax(angular_power_spec)
            DoA = self.doa_list[DoA_index]

        elif method == "periodic_ml":
            weighted_exp = np.mean(angular_power_spec * np.exp(1j * self.doa_list))
            DoA = np.angle(weighted_exp)

        elif method == "trimmed_periodic_ml":
            DoA_index = np.argmax(angular_power_spec)
            num_DoA = len(self.doa_list) // 2

            DoA_range = np.arange(-num_DoA // 2, num_DoA // 2 + 1) - DoA_index

            weighted_exp = np.mean(
                angular_power_spec[DoA_range] * np.exp(1j * self.doa_list[DoA_range])
            )
            DoA = np.angle(weighted_exp)

        else:
            raise NotImplementedError("this method is not yet implemented!")

        return DoA

    def run_demo(self):
        """
        This function runs the demo and shows the localization results.
        """
        # build a simple recorder
        mic = AudioRecorder()
        num_bits = 32

        # build a simple visualizer
        buffer_size = 60
        dim_samples = 1
        vz = Visualizer(
            buffer_size=buffer_size,
            dim_samples=dim_samples,
            waiting_time=2,
        )

        vz.start(
            figsize=(16, 10),
            xlabel="time",
            ylabel="DoA of the incoming audio",
            title=f"MUSIC: DoA estimation using multi-mic devkit with a circular array with 7 mics: fs:{self.fs} Hz, frame:{self.recording_duration} sec",
            grid=True,
        )

        while True:
            # get the new data from microphones
            data = mic.record_file(
                duration=self.recording_duration,
                bits=num_bits,
                fs=self.fs,
            )

            print("dimension of the input data: ", data.shape)

            # compute the maximum value in the data and use it as a threshold for beamforming
            ii_data = np.iinfo(data.dtype)
            max_value = ii_data.max

            rel_threshold = 0.0001
            threshold = rel_threshold * max_value

            # convert data into float format to avoid issues
            # remove the last channel and then apply beamforming
            # Note: the last channel contains always 0 in multi-mic devkit we were using
            data = np.asarray(data[:, :-1], dtype=np.float64)

            # recorded data information
            T, num_chan = data.shape
            time_vec = np.arange(0, T) / self.fs

            # do activity detection and stop the demo when there is no signal
            power_rec = np.sqrt(np.mean(data**2))

            print("received power from various microphones: ", power_rec)
            print("maximum value of the audio: ", max_value)
            print("threshold used for activity detection: ", threshold)

            if power_rec < threshold:
                # there is no activity
                vz.push(np.nan)
                print("signal is weak: no activity was detected!")

            else:
                # there is activity

                # compute the angular power spectrum of the input signal
                ang_pow_spec = self.music.beamforming(
                    sig_in=data,
                    num_active_freq=self.num_active_freq,
                    num_fft_bin=self.num_fft_bin,
                )

                # simplest method for estimating DoA
                method_list = ["peak", "periodic_ml", "trimmed_periodic_ml"]
                method = method_list[0]

                print("\n\n")
                print(
                    f"method used for DoA estimation from spike rate in `run_demo`: ",
                    method,
                )

                DoA = self.estimate_doa(angular_power_spec=ang_pow_spec, method=method)
                DoA_degree = DoA / np.pi * 180

                # push it to the visualizer
                vz.push(DoA_degree)


def run_demo():
    """
    this function runs the demo based on MUSIC and visualizes the DoA estimation.
    """

    # array geometry
    num_mic = 7
    radius = 4.5e-2

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # frequency range
    freq_range = np.asarray([1200, 2000])
    num_active_freq = 100

    # grid of DoAs
    num_grid = 32 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    # duration of recording in each section
    frame_duration = 0.25
    num_fft_bin = 2048
    fs = 48_000

    demo = Demo(
        geometry=geometry,
        freq_range=freq_range,
        num_active_freq=num_active_freq,
        doa_list=doa_list,
        recording_duration=frame_duration,
        num_fft_bin=num_fft_bin,
        fs=fs,
    )

    demo.run_demo()


def main():
    run_demo()


if __name__ == "__main__":
    main()
