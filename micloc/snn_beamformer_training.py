# ----------------------------------------------------------------------------------------------------------------------
# This module uses the beamforming designed for SNN and trains the SNN recursively to obtain the effect of STHT and
# analystic signal computation.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 15.09.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
from micloc.array_geometry import ArrayGeometry
from micloc.spike_encoder import SpikeEncoder
from typing import List
from rockpool.nn.modules.torch import LIFTorch, Linear
from rockpool.nn.combinators import Sequential
from numbers import Number
from tqdm import tqdm
import matplotlib.pyplot as plt

class SpikeDataset:
    def __init__(self, geometry: ArrayGeometry, spike_encoder: SpikeEncoder, doa_list: List[float], freq_range: List[float], size:int, duration: float,  fs:float):
        """
        This class builds a spike dataset for training RSNNs used for localization.
        Args:
            geometry (ArrayGeometry): geometry of the array.
            spike_encoder (SpikeEncoder): spike encoder used for converting input signals into spikes.
            doa_list (List[float]): a list containing target DoAs (as labels used for training).
            freq_range (List[float]): a list containing the minimim and maximum frequency of the spectrum of the signal.
            size (int): number of samples in the dataset.
            duration (float): duration of the samples.
            fs (float): sampling frequency of the signal.
        """
        self.geometry = geometry
        self.spike_encoder = spike_encoder
        self.doa_list = doa_list
        self.freq_range = freq_range
        self.size = size
        self.duration = duration
        self.fs = fs

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # produce a random angle
        rand_index = int(len(self.doa_list) * np.random.rand())
        doa = self.doa_list[rand_index]

        # compute delays
        delays = self.geometry.delays(doa)

        # produce a random frequency
        f_min, f_max = self.freq_range
        freq = f_min + (f_max - f_min) * np.random.rand()[0]
        time_vec = np.arange(0, self.duration, step=1/self.fs)

        sig_in = np.sin(2*np.pi * freq * (time_vec.reshape(1,-1) - delays.reshape(-1,1)))

        # convert into spikes
        spikes = self.spike_encoder.evolve(sig_in)

        return spikes

    def __iter__(self):
        self.start_index = -1
        return self

    def __next__(self):
        self.start_index += 1
        if self.start_index >= len(self):
            raise StopIteration

        return self[self.start_index]



class RSNNBeamforming:
    def __init__(self, beamforming_vecs:np.ndarray, dataset):
        self.beamforming_vecs = beamforming_vecs
        self.dataset = dataset

        self.num_mic, self.num_doa = beamforming_vecs.shape

        self.weight = np.zeros(self.num_mic, 2*self.num_doa)
        self.weight[:,::2] = np.real(beamforming_vecs)
        self.weight[:,1::2] = np.imag(beamforming_vecs)


    def evolve(self, num_epochs: int):
        """
        This function builds and trains a recursive SNN for training.

        Args:
            num_epochs (int): number of epochs in training.

        Returns:

        """
        dt = 1/self.fs

        tau_mem = 100 * dt
        tau_syn = 100 * dt

        num_neuron = 2 * self.num_doa

        W_rec = np.zeros(num_neuron)

        net = Sequential(
            Linear(
                shape=(self.num_mic, num_neuron),
                weight=self.weight,
                has_bias=False,
            ),
            LIFTorch(
                shape=(num_neuron, ),
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                W_rec=W_rec,
                dt=dt,
            )
        )

        print(net)



