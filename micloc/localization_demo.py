# ----------------------------------------------------------------------------------------------------------------------
# This module builds a simple visualization demo for multi-mic devkit.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.07.2023
# ----------------------------------------------------------------------------------------------------------------------
from micloc.record import AudioRecorder
from micloc.visualizer import Visualizer

from micloc.beamformer import Beamformer
from micloc.array_geometry import ArrayGeometry
from scipy.signal import butter, lfilter
import numpy as np


class Demo:
    def __init__(self, geometry: ArrayGeometry, freq_band: np.ndarray, doa_list: np.ndarray, recording_duration: float,
                 kernel_duration: float, fs: float):
        """
        this module builds a simple beamformer based localization demo.
        Args:
            geometry (ArrayGeometry): geometry of the array.
            freq_band (np.ndarray): an array containing the frequency range of the signal processed by the array.
            doa_list (np.ndarray): an array containing a grid of DoAs to be covered during localization.
            recording_duration (float): duration of each pack of recording.
            kernel_duration (float): duration of Hilbert kernel used for beamforming.
            fs (float): sampling period of the board.
        """

        # build the bemformer module
        self.beamf = Beamformer(geometry=geometry, kernel_duration=kernel_duration, fs=fs)

        # build a bandpass filter for the audio signal
        order = 1
        b, a = butter(order, freq_band, btype="bandpass", output='ba', fs=fs)
        self.filt = (b, a)

        self.doa_list = doa_list
        self.recording_duration = recording_duration
        self.kernel_duration = kernel_duration

        self.fs = fs

        # build the beamforming matrices
        num_samples = int(recording_duration * fs)
        noise = np.random.randn(num_samples)

        sig_in = lfilter(self.filt[0], self.filt[1], noise)
        sig_in = sig_in / np.max(np.abs(sig_in))

        time_in = np.arange(num_samples) / self.fs

        self.bf_mat = self.beamf.design_from_template(
            template=(time_in, sig_in),
            doa_list=self.doa_list,
        )

    def run(self):
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

        vz.start(figsize=(16, 10), xlabel="time", ylabel="power of voice",
                 title="DoA estimation using multi-mic devkit with a circular array with 7 mics",
                 grid=True)

        while True:
            # get the new data from microphones
            data = mic.record_file(
                duration=self.recording_duration,
                bits=num_bits,
                fs=self.fs,
            )

            # compute the maximum value in the data and use it as a threshold for beamforming
            ii_data = np.iinfo(data.dtype)
            max_value = ii_data.max

            rel_threshold = 0.0001
            threshold = rel_threshold * max_value

            # convert data into float format to avoid issues
            # remove the last channel and then apply beamforming
            # Note: the last channel contains always 0
            data = np.asarray(data[:, :-1], dtype=np.float64)

            power_rec = np.sqrt(np.mean(data ** 2))

            print("received power from various microphones: ", power_rec)
            print("maximum value of the audio: ", max_value)
            print("threshold used for activity detection: ", threshold)

            if power_rec < threshold:
                # there is no activity
                vz.push(np.nan)

            else:
                # there is activity

                # apply filtering to the selected frequency band
                data_filt = lfilter(self.filt[0], self.filt[1], data, axis=0)

                # apply beamforming
                data_bf = self.beamf.apply_to_signal(bf_mat=self.bf_mat, data_in=data_filt.T)

                # compute the active DoA
                power_grid = np.mean(np.abs(data_bf) ** 2, axis=1)

                DoA_index = np.argmax(power_grid)
                DoA = self.doa_list[DoA_index] * 180 / np.pi

                vz.push(DoA)


def test_demo():
    # array geometry
    num_mic = 7
    radius = 4.5e-2
    r_vec = radius * np.ones(num_mic)
    theta_vec = np.linspace(0, 2 * np.pi, num_mic)

    geometry = ArrayGeometry(r_vec=r_vec, theta_vec=theta_vec)

    # frequency range
    freq_band = np.asarray([1200, 1500])

    # grid of DoAs
    num_grid = 16 * num_mic
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    # duration of recording in each section
    recording_duration = 0.5
    fs = 48_000
    kernel_duration = 40e-3

    # build the demo
    demo = Demo(
        geometry=geometry,
        freq_band=freq_band,
        doa_list=doa_list,
        recording_duration=recording_duration,
        kernel_duration=kernel_duration,
        fs=fs
    )

    # run the demo
    demo.run()


def main():
    test_demo()


if __name__ == '__main__':
    main()
