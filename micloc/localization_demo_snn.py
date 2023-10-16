# ----------------------------------------------------------------------------------------------------------------------
# This module builds a simple visualization demo for multi-mic devkit.
#
# Note: in this version, we are using the final SNN version for target localization.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 16.10.2023
# ----------------------------------------------------------------------------------------------------------------------
from archive.record import AudioRecorder
from micloc.visualizer import Visualizer

from micloc.snn_beamformer import SNNBeamformer
from micloc.array_geometry import ArrayGeometry, CenterCircularArray
from micloc.filterbank import ButterworthFilterbank
import numpy as np


class Demo:
    def __init__(self, geometry: ArrayGeometry, freq_bands: np.ndarray, doa_list: np.ndarray, recording_duration: float,
                 kernel_duration: float, fs: float):
        """
        this module builds an SNN beamformer based localization demo.
        Args:
            geometry (ArrayGeometry): geometry of the array.
            freq_bands (np.ndarray): an array of dimension C x 2 whose rows specify the frequency range of the signal processed by the array in various frequency channels.
            doa_list (np.ndarray): an array containing a grid of DoAs to be covered during localization.
            recording_duration (float): duration of each pack of recording.
            kernel_duration (float): duration of Hilbert kernel used for beamforming.
            fs (float): sampling period of the board.
        """

        # build the beamformer module
        # since we may have targeted several frequency bands we need several beamforming modules
        freq_bands = np.asarray(freq_bands)

        if freq_bands.ndim == 1:
            # there is only a single band to cover
            freq_bands = freq_bands.reshape(1,-1)

        # beamforming modules and corresponding beamforming matrices
        self.beamfs = []
        self.bf_mats = []

        # target several frequency bands
        for freq_range in freq_bands:
            # use the center frequency as the reference frequency
            freq_mid = np.mean(freq_range)

            # time constants
            tau_mem = 1/(2*np.pi*freq_mid)
            tau_syn = tau_mem
            tau_vec = [tau_syn, tau_mem]

            # build SNN beamforming module
            beamf = SNNBeamformer(geometry=geometry, kernel_duration=kernel_duration, freq_range=freq_range, tau_vec=tau_vec, fs=fs)
            self.beamfs.append(beamf)

            # build the template signal and design bemforming vectors
            time_temp = np.arange(0, recording_duration, step=1/fs)
            sig_temp = np.sin(2*np.pi*freq_mid * time_temp)

            bf_vecs = beamf.design_from_template(template=(time_temp, sig_temp), doa_list=doa_list)

            self.bf_mats.append(bf_vecs)

        # build a filterbank for various frequency bands covered by the array
        order = 1
        self.filterbank = ButterworthFilterbank(freq_bands=freq_bands, order=order, fs=fs)

        self.doa_list = np.asarray(doa_list)
        self.recording_duration = recording_duration
        self.kernel_duration = kernel_duration

        self.fs = fs

    def run(self):
        """
        This function runs the demo and shows the localziation results.
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

        vz.start(figsize=(16, 10), xlabel="time", ylabel="DoA of the incoming audio",
                 title=f"DoA estimation using multi-mic devkit with a circular array with 7 mics",
                 grid=True
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
            time_vec = np.arange(0, T)/self.fs

            # do activity detection and stop the demo when there is no signal
            power_rec = np.sqrt(np.mean(data ** 2))

            print("received power from various microphones: ", power_rec)
            print("maximum value of the audio: ", max_value)
            print("threshold used for activity detection: ", threshold)

            if power_rec < threshold:
                # there is no activity
                vz.push(np.nan)
                print("signal is weak: no activity was detected!")

            else:
                # there is activity

                # apply filterbank to the selected frequency band
                data_filt = self.filterbank.evolve(sig_in=data)

                print("dimension of filterbank output: ", data_filt.shape)

                # apply beamforming to various channels and add the power
                power_grid = 0

                for data_filt_chan, bf_mat_chan, beamf_module in zip(data_filt, self.bf_mats, self.beamfs):
                    # compute the beamformed signal in each frequency channel separately
                    data_bf_chan = beamf_module.apply_to_signal(bf_mat=bf_mat_chan, sig_in_vec=(time_vec, data_filt_chan))

                    # compute the power vs. DoA (power spectrum) after beamforming
                    power_grid_chan = np.mean(np.abs(data_bf_chan) ** 2, axis=0)

                    # accumulate the power received from various frequency channels to obtain the final angular pattern
                    power_grid = power_grid + power_grid_chan

                # compute the DoA of the strongest target
                DoA_index = np.argmax(power_grid)
                DoA = self.doa_list[DoA_index] * 180 / np.pi

                # push it to the visualizer
                vz.push(DoA)


def test_demo():
    # array geometry
    num_mic = 7
    radius = 4.5e-2

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # frequency range
    freq_bands = [
        [1600, 2400],
    ]


    # grid of DoAs
    num_grid = 16 * num_mic
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    # duration of recording in each section
    recording_duration = 0.25
    fs = 48_000
    kernel_duration = 10e-3

    # build the demo
    demo = Demo(
        geometry=geometry,
        freq_bands=freq_bands,
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
