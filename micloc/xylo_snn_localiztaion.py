# ----------------------------------------------------------------------------------------------------------------------
# This module builds a simple visualization demo for multi-mic devkit.
# It uses the Xylo chip to process the spikes and used the resulting rate encoding to localize and track targets.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 17.10.2023
# ----------------------------------------------------------------------------------------------------------------------
from re import S
from archive.record import AudioRecorder
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
from rockpool.devices.xylo.syns61201 import XyloSamna, config_from_specification, xa2_devkit_utils as hdu, mapper
from rockpool.devices.xylo.syns61201 import XyloSim

from rockpool.transform import quantize_methods as q


from scipy.signal import lfilter
import matplotlib.pyplot as plt
import time


class Demo:
    def __init__(self, geometry: ArrayGeometry, freq_bands: np.ndarray, doa_list: np.ndarray, recording_duration: float,
                 kernel_duration: float, bipolar_spikes: bool, fs: float):
        """
        this module builds an SNN beamformer based localization demo.
        Args:
            geometry (ArrayGeometry): geometry of the array.
            freq_bands (np.ndarray): an array of dimension C x 2 whose rows specify the frequency range of the signal processed by the array in various frequency channels.
            doa_list (np.ndarray): an array containing a grid of DoAs to be covered during localization.
            recording_duration (float): duration of each pack of recording.
            kernel_duration (float): duration of Hilbert kernel used for beamforming.
            bipolar_spikes (bool): if bipolar spike encoding is used for localization. Defaults to False.
            fs (float): sampling period of the board.
        """

        # build the beamformer module
        # since we may have targeted several frequency bands we need several beamforming modules
        self.freq_bands = np.asarray(freq_bands)

        if self.freq_bands.ndim == 1:
            # there is only a single band to cover
            self.freq_bands = self.freq_bands.reshape(1,-1)

        # beamforming modules and corresponding beamforming matrices
        self.beamfs = []
        self.bf_mats = []
        self.tau_vecs = []

        # target several frequency bands
        for freq_range in self.freq_bands:
            # use the center frequency as the reference frequency
            freq_mid = np.mean(freq_range)

            # time constants
            tau_mem = 1/(2*np.pi*freq_mid)
            tau_syn = tau_mem
            tau_vec = [tau_syn, tau_mem]

            self.tau_vecs.append(tau_vec)

            # build SNN beamforming module
            beamf = SNNBeamformer(geometry=geometry, kernel_duration=kernel_duration, freq_range=freq_range, tau_vec=tau_vec, bipolar_spikes=bipolar_spikes, fs=fs)
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
        self.dt = 1.0/self.fs

        self._initialize_snn_module()

    
    def _initialize_snn_module(self):
        """ this module initializes xylo-a2 board for SNN processing and localization """
        
        print("\n")
        print("+"*150)
        print(" trying to connect to Xylo-a2 devkit ".center(150,"+"))
        print("+"*150)

        # compute the number of input channels
        num_freq_chan = len(self.freq_bands)
        spike_dim_in_chan, spike_dim_out_chan = self.bf_mats[0].shape

        num_ch_in = num_freq_chan * spike_dim_in_chan
        num_ch_out = num_freq_chan * spike_dim_out_chan

        # weight connection between layers
        weight = np.zeros((num_ch_in, num_ch_out))
        for ch in range(num_freq_chan):
            weight[ch*spike_dim_in_chan : (ch+1)*spike_dim_in_chan, ch*spike_dim_out_chan:(ch+1)*spike_dim_out_chan] = self.bf_mats[ch]

        weight = torch.tensor(data=weight, dtype=torch.float32)

        # thresholds
        # TODO: we need to adjust the threshold based on the activity in various frequency channels
        # TODO: how should this be done?
        threshold = 1
        thresholds = threshold * torch.ones(num_ch_out)

        tau_mem_vec = []
        tau_syn_vec = []
        for tau_syn, tau_mem in self.tau_vecs:
            tau_syn_vec.extend([tau_syn for _ in range(spike_dim_out_chan)])
            tau_mem_vec.extend([tau_mem for _ in range(spike_dim_out_chan)])
        
        # extra factor
        extra_factor = 10
        tau_mem_vec = extra_factor * torch.tensor(data=tau_mem_vec, dtype=torch.float32)
        tau_syn_vec = extra_factor * torch.tensor(data=tau_syn_vec, dtype=torch.float32)


        # build the network
        # NOTE: we add a dummy node at the end to make sure that we can deploy the netowrk and read
        # hidden layer outputs as canidates for rate encoding. 
        # NOTE: the number of neurons in hidden layer is equal to the number of grid points in DoA estimation x number of frequency channels.
        threshold = 0.2

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
                dt=self.dt,
            ),
            LinearTorch(
                shape=(num_ch_out, 1),
                weight=torch.zeros(num_ch_out, 1),
                has_bias=False,
            ),
            LIFTorch(
                shape=(1,),
                threshold=1000,
                tau_syn=tau_syn_vec[0],
                tau_mem=tau_mem_vec[0],
                dt=self.dt,
            ),
        )

        # get the chip version of the network
        g = self.net.as_graph()

        # map the graph to Xylo HW architecture
        spec = mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')

        # quantize the parameters to Xylo HW constraints
        quant_spec = spec.copy()
        quant_spec.update(q.global_quantize(**quant_spec))

        # get the HW config that we can use on Xylosim
        xylo_conf, is_valid, message = config_from_specification(**quant_spec)

        print('Valid config: ', is_valid)

        # TODO: xylosim module did not work
        #XyloSim_model = XyloSim.from_config(xylo_conf, dt=self.dt)
        #spikes_out, _, _ = XyloSim_model(spikes, record=True)

        # build the xylo-samna version
        hdks = hdu.find_xylo_a2_boards()
        assert len(hdks) > 0, 'No Xylo-A2 found'
        hdk = hdks[0]
        print('Xylo a2 HW found')

        print(f'Clock freq. set to {hdu.set_xylo_core_clock_freq(hdk, 6.25)} MHz')

        config, is_valid, msg = config_from_specification(**quant_spec)
        print('config is valid',is_valid)

        self.xylo_a2 = XyloSamna(hdk, config)

        print("Xylo a2 device was initialized successfully!")

    

    
    def spike_encoding(self, sig_in: np.ndarray)-> np.ndarray:
        """this function processes the input signal received from microphone and produces the spike encoding to be applied to SNN.

        Args:
            sig_in (np.ndarray): input `T x num_mic` signal received from the microphones.

        Returns:
            np.ndarray: `T x (num_mic x 2 x num_freq_chan)` spike signal produced via spike encoding.
        """

        # apply STHT to produce the STHT transform
        # NOTE: all the frequency channels use the same STHT
        stht_kernel = self.beamfs[0].kernel

        sig_in_h = np.roll(sig_in, len(stht_kernel)//2, axis=0) + 1j * lfilter(stht_kernel, [1], sig_in, axis=0)

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

        return spikes_in

        
    def xylo_process(self, spikes_in: np.ndarray, record_power:bool=False) -> np.ndarray:
        """this function passes the spikes obtained from spike encoding of the input signal to the Xylo chip
        and returns the recorded spikes.

        Args:
            spikes_in (np.ndarray): input spikes of dimension `T x num_mic`.
            record_power (bool): record the consumed power in the chip. Defaults to False.

        Returns:
            np.ndarray: output spikes produced by the Xylo chip.
        """
        # reset the board
        self.xylo_a2.reset_state()

        # process the spikes with xylo_a2
        # NOTE: recoridng is needed because we have put a dummy spike at the end to satisfy the configuration
        # this is needed because the number of output channels in our case is much larger than the maximum 16
        # output channels permitted for xylo-a2
        record = True
        _ , _ , rec =  self.xylo_a2.evolve(spikes_in[:100,:], record=record, record_power=record_power)
        
        # no power measurement so we can skip this field
        # snn_power = rec['logic_power'].mean()
        # print('measured power of SNN core for current sample in mw:', snn_power *1000 )

        # find the intermediate spikes representing the first layer of neurons
        spikes_out = rec['Spikes']

        return spikes_out


    def extract_rate(self, spikes_in: np.ndarray)->np.ndarray:
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

        rate_DoA = rate_channels.reshape(-1, num_DoA_grid).mean(0) * self.fs

        return rate_DoA




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

                # process the input signal and produce spikes
                spikes_in = self.spike_encoding(sig_in=data)

                # process the spikes with xylo_a2 
                record_power = False

                # compute the accumulated spike rate from all channels
                start_xylo_a2_process = time.time()
                spike_rate = self.xylo_process(spikes_in=spikes_in, record_power=record_power)
                duration_xylo_a2_process = time.time() - start_xylo_a2_process

                print("duration of spike processing by xulo-a2 kit: ", duration_xylo_a2_process)
                
                # compute the DoA of the strongest target
                DoA_index = np.argmax(spike_rate)
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
        [1600, 1700],
    ]


    # grid of DoAs
    num_grid = 16 * num_mic
    doa_list = np.linspace(-np.pi, np.pi, num_grid)

    # duration of recording in each section
    recording_duration = 0.1
    fs = 48_000
    kernel_duration = 10e-3

    # build the demo
    bipolar_spikes = False
    demo = Demo(
        geometry=geometry,
        freq_bands=freq_bands,
        doa_list=doa_list,
        recording_duration=recording_duration,
        kernel_duration=kernel_duration,
        bipolar_spikes=bipolar_spikes,
        fs=fs
    )

    # run the demo
    demo.run()


def main():
    test_demo()


if __name__ == '__main__':
    main()
