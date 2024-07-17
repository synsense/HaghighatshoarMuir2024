# ----------------------------------------------------------------------------------------------------------------------
# This module builds a simple visualization demo for multi-mic devkit.
# It uses the XyloSim model to process the spikes as it happens within the chip.
# The resulting rate encoding is used to do localization and track the targets.
# 
# Output of the demo is used to do some benchmarking on the SNN localization performance in real-time.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 17.07.2024
# ----------------------------------------------------------------------------------------------------------------------
from micloc.record import AudioRecorder
from micloc.visualizer import Visualizer

from micloc.snn_beamformer import SNNBeamformer
from micloc.array_geometry import ArrayGeometry, CenterCircularArray
from micloc.filterbank import ButterworthFilterbank
import numpy as np

# rockpool module for deployment into Xylo
from rockpool.nn.modules import LinearTorch, LIFBitshiftTorch, LIFTorch
from rockpool.nn.combinators import Sequential
import torch

# rockpool conversion/mapping modules
# from rockpool.devices.xylo.syns61201 import (
#     config_from_specification,
#     mapper,
#     xa2_devkit_utils as hdu,
#     XyloSamna,
# )
from rockpool.devices.xylo.syns61201 import XyloSim, mapper, config_from_specification

from rockpool.transform import quantize_methods as q

from scipy.signal import lfilter
import time
import matplotlib.pyplot as plt

import os
from pathlib import Path


class DemoBenchmark:
    def __init__(
        self,
        geometry: ArrayGeometry,
        freq_bands: np.ndarray,
        doa_list: np.ndarray,
        recording_duration: float = 0.25,
        kernel_duration: float = 10e-3,
        bipolar_spikes: bool = True,
        fs: float = 48_000,
    ):
        """
        this module builds an SNN beamformer based localization demo.
        the output of the demo is then used to do a real-time benchmark of the performance of the SNN localization algorithm.

        Args:
            geometry (ArrayGeometry): geometry of the array.
            freq_bands (np.ndarray): an array of dimension C x 2 whose rows specify the frequency range of the signal processed by the array in various frequency channels.
            doa_list (np.ndarray): an array containing a grid of DoAs to be covered during localization.
            recording_duration (float): duration of each pack of recording. Deafults to 0.25 sec (250 ms).
            kernel_duration (float): duration of Hilbert kernel used for beamforming. Defaults to 10ms.
            bipolar_spikes (bool): if bipolar spike encoding is used for localization. Defaults to True.
            NOTE: the hardware version can be quite slow in the case of localization. Defaults to 48_000 Hz for the multi-mic board.

            fs (float): sampling period of the board.
        """

        # build the beamformer module
        # since we may have targeted several frequency bands we need several beamforming modules
        self.freq_bands = np.asarray(freq_bands)

        if self.freq_bands.ndim == 1:
            # there is only a single band to cover
            self.freq_bands = self.freq_bands.reshape(1, -1)

        # beamforming modules and corresponding beamforming matrices
        self.beamfs = []
        self.bf_mats = []
        self.tau_vecs = []

        # target several frequency bands
        for freq_range in self.freq_bands:
            # use the center frequency as the reference frequency
            freq_mid = np.mean(freq_range)

            # time constants
            tau_mem = 1 / (2 * np.pi * freq_mid)
            tau_syn = tau_mem
            tau_vec = [tau_syn, tau_mem]

            self.tau_vecs.append(tau_vec)

            # build SNN beamforming module
            beamf = SNNBeamformer(
                geometry=geometry,
                kernel_duration=kernel_duration,
                freq_range=freq_range,
                tau_vec=tau_vec,
                bipolar_spikes=bipolar_spikes,
                fs=fs,
            )
            self.beamfs.append(beamf)

            # build the template signal and design bemforming vectors
            time_temp = np.arange(0, recording_duration, step=1 / fs)
            sig_temp = np.sin(2 * np.pi * freq_mid * time_temp)

            bf_vecs = beamf.design_from_template(
                template=(time_temp, sig_temp), doa_list=doa_list
            )

            self.bf_mats.append(bf_vecs)

        self.tau_vecs = np.asarray(self.tau_vecs)

        # build a filterbank for various frequency bands covered by the array
        order = 1
        self.filterbank = ButterworthFilterbank(
            freq_bands=freq_bands, order=order, fs=fs
        )

        self.doa_list = np.asarray(doa_list)
        self.recording_duration = recording_duration
        self.kernel_duration = kernel_duration

        # are spikes bipolar?
        self.bipolar_spikes = bipolar_spikes

        self.fs = fs
        self.dt = 1.0 / self.fs

        # this change of timing is needed in chip version to let it work but not necessary in XyloSim version although
        # it is also ok since there is no time in XyloSim and as far as scaling is concerned, we have no issue!
        target_dt = 1.0e-3
        self._initialize_snn_module(target_dt=target_dt)

    def _initialize_snn_module(self, target_dt: float):
        """this module initializes xyloxim for SNN processing and localization.

        Args:
            target_dt (float): `dt` used for simulation of SNN core.

            NOTE: change of dt is not needed in XyloSim since everything gets scaled in software.
            It is important in hardware version only where dt gives the period with which spikes are pushed into the chip.
        """

        # ===========================================================================
        #                    CHANGE and RESCALE all time constants
        # ===========================================================================
        target_fs = 1 / target_dt
        scale = self.fs / target_fs
        scaled_tau_vecs = np.copy(self.tau_vecs) * scale

        print("\n")
        print("+" * 150)
        print(" trying to configure xylosim ".center(150, "+"))
        print("+" * 150)

        # compute the number of input channels
        num_freq_chan = len(self.freq_bands)
        spike_dim_in_chan, spike_dim_out_chan = self.bf_mats[0].shape

        num_ch_in = num_freq_chan * spike_dim_in_chan
        num_ch_out = num_freq_chan * spike_dim_out_chan

        # weight connection between layers
        weight = np.zeros((num_ch_in, num_ch_out))
        for ch in range(num_freq_chan):
            weight[
                ch * spike_dim_in_chan : (ch + 1) * spike_dim_in_chan,
                ch * spike_dim_out_chan : (ch + 1) * spike_dim_out_chan,
            ] = self.bf_mats[ch]

        # consider the spike polarity effect
        if self.bipolar_spikes:
            # copy positive and negative version of weights
            weight = np.vstack([weight, -weight])

            # increase number iof input channels
            num_ch_in *= 2

        weight = torch.tensor(data=weight, dtype=torch.float32)

        tau_mem_vec = []
        tau_syn_vec = []
        for tau_syn, tau_mem in scaled_tau_vecs:
            tau_syn_vec.extend([tau_syn for _ in range(spike_dim_out_chan)])
            tau_mem_vec.extend([tau_mem for _ in range(spike_dim_out_chan)])

        # extra factor
        tau_mem_vec = torch.tensor(data=tau_mem_vec, dtype=torch.float32)
        tau_syn_vec = torch.tensor(data=tau_syn_vec, dtype=torch.float32)

        # add recurrnet weights to neurons to make sure that the DC value of the membrane potential is set to zero
        w_rec_coef = -0.1 / num_ch_out
        w_rec = w_rec_coef * torch.ones((num_ch_out, num_ch_out), dtype=torch.float32)

        # build the network NOTE: we add a dummy node at the end to make sure that we can deploy the netowrk and read
        # hidden layer outputs as canidates for rate encoding. NOTE: the number of neurons in hidden layer is equal
        # to the number of grid points in DoA estimation x number of frequency channels.
        threshold = 1.0

        self.net = Sequential(
            LinearTorch(
                shape=(num_ch_in, num_ch_out),
                weight=weight,
                has_bias=False,
            ),
            LIFTorch(
                shape=(num_ch_out,),
                threshold=threshold,
                tau_syn=tau_syn_vec,
                tau_mem=tau_mem_vec,
                has_rec=True,
                w_rec=w_rec,
                dt=target_dt,
            ),
            LinearTorch(
                shape=(num_ch_out, 1),
                weight=torch.ones(num_ch_out, 1),
                has_bias=False,
            ),
            LIFTorch(
                shape=(1,),
                threshold=1,
                tau_syn=tau_syn_vec[0],
                tau_mem=tau_mem_vec[0],
                dt=target_dt,
            ),
        )

        # map the graph to Xylo HW architecture
        spec = mapper(
            self.net.as_graph(),
            weight_dtype="float",
            threshold_dtype="float",
            dash_dtype="float",
        )

        # quantize the parameters to Xylo HW constraints
        spec.update(q.global_quantize(**spec))

        # get the HW config that we can use on Xylosim
        xylo_config, is_valid, message = config_from_specification(**spec)

        if is_valid:
            print("configuration is valid!")
            print(message)

        # build simulation module: xylosim or hardware version
        self.xylo = XyloSim.from_config(
            xylo_config, output_mode="Spike", dt=target_dt
        )

    def spike_encoding(self, sig_in: np.ndarray) -> np.ndarray:
        """this function processes the input signal received from microphone and produces the spike encoding to be applied to SNN.

        Args:
            sig_in (np.ndarray): input `T x num_mic` signal received from the microphones.

        Returns:
            np.ndarray: `T x (num_mic x 2 x num_freq_chan)` spike signal produced via spike encoding.
        """

        # apply STHT to produce the STHT transform
        # NOTE: all the frequency channels use the same STHT
        stht_kernel = self.beamfs[0].kernel

        sig_in_h = np.roll(sig_in, len(stht_kernel) // 2, axis=0) + 1j * lfilter(
            stht_kernel, [1], sig_in, axis=0
        )

        # real-valued version of the signal
        sig_in_real = np.hstack([np.real(sig_in_h), np.imag(sig_in_h)])

        # apply filters in the filterbank to further decompose the signal
        sig_in_real_filt = self.filterbank.evolve(sig_in=sig_in_real)

        # join all the frequency channels
        # NOTE: num_chan = 2 x num_mic
        F, T, num_chan = sig_in_real_filt.shape
        sig_in_all = np.hstack([sig_in_real_filt[ch] for ch in range(F)])

        # compute spike encoding
        # NOTE: we also need to convert the spikes into integer format
        spk_encoder = self.beamfs[0].spk_encoder
        spikes_in = spk_encoder.evolve(sig_in_all).astype(np.int64)

        # modify the spikes based on polarity
        if self.bipolar_spikes:
            spikes_pos = ((spikes_in + np.abs(spikes_in)) / 2).astype(np.int64)
            spikes_neg = ((-spikes_in + np.abs(spikes_in)) / 2).astype(np.int64)

            spikes_in = np.hstack([spikes_pos, spikes_neg])

        return spikes_in

    def xylo_process(self, spikes_in: np.ndarray) -> np.ndarray:
        """this function passes the spikes obtained from spike encoding of the input signal to the Xylo chip
        and returns the recorded spikes.

        Args:
            spikes_in (np.ndarray): input spikes of dimension `T x num_mic`.

        Returns:
            np.ndarray: output spikes produced by the Xylo chip.
        """

        # process the spikes with xylosim
        self.xylo.reset_state()

        out, state, rec = self.xylo(spikes_in, record=True)

        # find the intermediate spikes representing the first layer of neurons
        spikes_out = rec["Spikes"]

        return spikes_out

    def extract_rate(self, spikes_in: np.ndarray) -> np.ndarray:
        """this module processes the collected spikes from xylo_a2 and extracts the spikes rate for DoA grids.

        Args:
            spikes_in (np.ndarray): spikes produced by xylo-a2.

        Returns:
            np.ndarray: spike rate at various DoA grid channels.
        """
        num_freq_channels = len(self.freq_bands)
        num_DoA_grid = len(self.doa_list)

        # reshape `T x (num_freq_channels x num_DoA_grid)` into `num_freq_channles` signals each of shape `T x num_DoA_grid`
        # and aggregate their spike rate as a measure of their power or strength

        rate_channels = np.mean(spikes_in, axis=0) * self.fs

        rate_DoA = rate_channels.reshape(-1, num_DoA_grid).mean(0)

        return rate_DoA

    def estimate_doa_from_rate(self, spike_rate: np.ndarray, method: str) -> float:
        """this method allows to estimate the DoA from the spike rate pattern.
        NOTE: each method may be good for one scenario but may not be good when there are several targets.
        This should be known a priori.

        Args:
            spike_rate (np.ndarray): average spike rate obtained from XyloSim version.
            method (str): name of the method. Options are:
                - "peak" : use the DoA corresponding to the peak value of spike rate.
                - "periodic_ml" : this method treats DoA as a periodic function and applies ML estimate to it.
                - "trimmed_periodic_ml" : the method chooses the spike rates around the peak value and estimates DoA using periodic ML method.

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
            DoA_index = np.argmax(spike_rate)
            DoA = self.doa_list[DoA_index]

        elif method == "periodic_ml":
            weighted_exp = np.mean(spike_rate * np.exp(1j * self.doa_list))
            DoA = np.angle(weighted_exp)

        elif method == "trimmed_periodic_ml":
            DoA_index = np.argmax(spike_rate)
            num_DoA = len(self.doa_list) // 2

            DoA_range = np.arange(-num_DoA // 2, num_DoA // 2 + 1) - DoA_index

            weighted_exp = np.mean(
                spike_rate[DoA_range] * np.exp(1j * self.doa_list[DoA_range])
            )
            DoA = np.angle(weighted_exp)

        else:
            raise NotImplementedError("this method is not yet implemented!")

        return DoA

    def benchmark(self, num_samples: int):
        """
        This function runs the demo and benchmarks the SNN localization performance using real-time data.

        Args:
            num_samples (int): number of DoA estimation samples to be recorded.
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
    
        # data to be collected for benchmark
        doa_estimate = []
        num_collected_samples = 0

        vz.start(
            figsize=(16, 10),
            xlabel="time",
            ylabel="DoA of the incoming audio",
            title=f"multi-mic snn localization: circular array with 7 mics: fs:{self.fs} Hz, frame:{self.recording_duration} s, kernel:{int(1000 * self.kernel_duration)} ms, bipolar-spike:{self.bipolar_spikes}",
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

                # process the input signal and produce spikes
                spikes_in = self.spike_encoding(sig_in=data)

                # compute the accumulated spike rate from all channels
                start_xylosim_process = time.time()
                spikes_out = self.xylo_process(spikes_in=spikes_in)
                duration_xylosim_process = time.time() - start_xylosim_process

                print(
                    "duration of spike processing by xylosim: ",
                    duration_xylosim_process,
                )

                # compute the DoA of the strongest target
                spike_rate = self.extract_rate(spikes_out)

                # simplest method for estimating DoA
                method_list = ["peak", "periodic_ml", "trimmed_periodic_ml"]
                method = method_list[0]

                print("\n\n")
                print(
                    f"method used for DoA estimation from spike rate in `run_demo`: ",
                    method,
                )

                DoA = self.estimate_doa_from_rate(spike_rate=spike_rate, method=method)
                DoA_degree = DoA / np.pi * 180

                # push it to the visualizer
                vz.push(DoA_degree)

                # record the data as well
                doa_estimate.append(DoA_degree)

                num_collected_samples += 1

                if num_collected_samples >= num_samples:
                    vz.stop()
                    break

        return np.asarray(doa_estimate)

def benchmark(num_samples: int, frame_duration: float, filename:str):
    """
    this function runs the demo based on SNN and uses the estimated DoA to benchmark the localization performance.

    Args:
        num_samples (int): number of DoA samples to be collected for benchmarking.
        frame_duration(float): duration of each frame of data over which a single DoA estimation is done.
        filename (str): where to store the collected data.
    """

    # array geometry
    num_mic = 7
    radius = 4.5e-2

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic)

    # frequency range
    freq_bands = [
        [1600, 1900],
        # [2700, 3000],
        # [3700, 4000]
    ]

    # grid of DoAs
    num_grid = 64 * num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    # duration of recording in each section
    # recording_duration = 0.25
    recording_duration = frame_duration
    fs = 48_000
    kernel_duration = 10e-3

    # build the demo
    bipolar_spikes = True

    demo = DemoBenchmark(
        geometry=geometry,
        freq_bands=freq_bands,
        doa_list=doa_list,
        recording_duration=recording_duration,
        kernel_duration=kernel_duration,
        bipolar_spikes=bipolar_spikes,
        fs=fs,
    )

    doa_samples = demo.benchmark(num_samples)
    print("collected doa-sampels: ", doa_samples)


    np.savetxt(
        fname=filename,
        X=doa_samples,
        fmt="%0.6f",
    )


    # plot the histogram of results
    benchmark_duration = num_samples * frame_duration

    plt.close()
    plt.hist(doa_samples, num_grid)
    plt.xlabel("DoA")
    plt.ylabel("histogram")
    plt.title("SNN localization performance for {} second".format(benchmark_duration))

def analyze(filename: str):
    """this file reads the recorded data and does statistical analysis.

    Args:
        filename (str): name of the file.
    """
    data = np.loadtxt(
        fname=filename
    )

    data = np.asarray(data)

    doa_mean = np.mean(data)
    doa_std = np.std(data)
    doa_med = np.median(data)
    doa_mad = np.median(np.abs(data - doa_med))

    print("mean doa: ", doa_mean)
    print("std: ", doa_std)

    print("doa median: ", doa_med)
    print("doa mean abs deviation: ", doa_mad)

    # NOTE: this is calculated based on the Gaussian distribution where E[|x|] is `sqrt(2/pi) * sigma`
    # so this yields a plug-in estimator for the value of `sigma` given by E[|x|] x sqrt(pi/2)
    print("robust std: ", doa_mad * np.sqrt(np.pi/2))

    num_mic = 7
    num_grid = 64 * num_mic + 1

    plt.hist(data, num_grid)
    plt.xlabel("DoA")
    plt.ylabel("histogram")
    plt.grid(True)
    plt.title("Histogram of collected DoA samples")
    plt.show()


def main():
    num_samples = 100
    frame_duration = 0.25

    # save the collected data]
    folder = Path(__file__).resolve().parent
    filename = os.path.join(folder, "demo_benchmark_data.txt")

    benchmark(
        num_samples=num_samples,
        frame_duration=frame_duration,
        filename=filename
    )

    # wait for the visualization to end completely
    time.sleep(3)
    analyze(filename=filename)

if __name__ == "__main__":
    main()
