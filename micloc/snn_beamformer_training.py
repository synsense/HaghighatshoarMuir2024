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
from operator import ge
from pickletools import optimize
import numpy as np
from micloc.array_geometry import ArrayGeometry, CenterCircularArray, CircularArray
from micloc.spike_encoder import SpikeEncoder, ZeroCrossingSpikeEncoder
from typing import List
from rockpool.nn.modules.torch import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam 
from numbers import Number
from tqdm import tqdm
from  mlflow import log_metric, log_param, log_params, log_artifacts
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time


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
        rand_index = int(len(self.doa_list) * np.random.rand(1)[0])
        doa = self.doa_list[rand_index]

        # compute delays
        delays = self.geometry.delays(doa)

        # produce a random frequency
        f_min, f_max = self.freq_range
        freq = f_min + (f_max - f_min) * np.random.rand(1)[0]
        time_vec = np.arange(0, self.duration, step=1/self.fs)

        sig_in = np.sin(2*np.pi * freq * (time_vec.reshape(1,-1) - delays.reshape(-1,1)))

        # convert into spikes
        spikes = self.spike_encoder.evolve(sig_in)

        # label of the data
        label = np.zeros(len(self.doa_list))
        label[rand_index] = 1

        return spikes, label

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

        self.weight = np.zeros((self.num_mic, 2*self.num_doa))
        self.weight[:,::2] = np.real(beamforming_vecs)
        self.weight[:,1::2] = np.imag(beamforming_vecs)

        self.dataset = dataset


    def evolve(self, num_epochs: int):
        """
        This function builds and trains a recursive SNN for training.

        Args:
            num_epochs (int): number of epochs in training.

        Returns:

        """
        # set the device for implementation
        device = torch.device("cuda")
        
        # network parameters
        dt = 1/self.dataset.fs

        tau_mem = 10 * dt
        tau_syn = 10 * dt
        threshold = 10

        num_neuron = 2 * self.num_doa
        num_mic = self.num_mic

        # recursive weight
        weight_rec = -0.1*torch.eye(num_neuron, num_neuron, dtype=torch.float64, device=device)
        odd_indices = np.arange(1, num_neuron, step=2)
        weight_rec[odd_indices, odd_indices] = 0

        # mask for w_rec
        mask_rec = np.kron(np.eye(self.num_doa), np.ones((2,2)))
        mask_rec = torch.tensor(data=mask_rec, dtype=torch.float64).to(device=device)

        weight = torch.tensor(data=self.weight, dtype=torch.float64)

        net = Sequential(
            LinearTorch(
                shape=(num_mic, num_neuron),
                weight=weight,
                has_bias=False,
            ),
            LIFTorch(
                shape=(num_neuron, ),
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                threshold=threshold,
                has_rec=True,
                w_rec=weight_rec,
                dt=dt,
            )
        )

        net = net.to(device=device)

        batch_size = 10
        shuffle = True
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

        # optimizer
        lr = 0.001
        optimizer = Adam(net.parameters().astorch(), lr=lr)

        for epoch, (data, label) in enumerate(tqdm(dataloader)):
            data = data.transpose(1,2).to(device)

            spikes, state, recording = net(data, record=True)

            # compute the probability of firing by averaging in time
            prob = spikes.mean(axis=1)
            
            # aggregate two in-phase and quadrature spikes
            prob = prob[:, ::2] + prob[:, 1::2]
            EPS = 0.001
            prob = prob/(prob.sum(axis=1).reshape(-1,1) + EPS)
            
            # log_metric("prob_angle", prob[0].detach().cpu().numpy())
            

            # loss = torch.log(1/prob[index])
            loss = torch.mean((prob[label==1] - 1.0)**2)
            
            if np.isnan(loss.item()):
                raise ValueError("None occured during the training! Training was stopped!")         
            
            loss.backward()
            log_metric("grad_w_rec_norm", torch.mean(net[1].w_rec.grad**2))
            log_metric("loss", loss.item())
            
            output_path = os.path.join(Path(__file__).resolve().parent.parent, "output")
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            
            log_artifacts(output_path)
            
            if (epoch + 4) % 4 == 0:
                output_filename = os.path.join(output_path, f"prob{time.asctime()}.txt".replace(" ", ""))
                data = prob.detach().cpu().numpy()[0]
                
                index = np.argsort(data)
                
                data = data[index]
                data_label = label.detach().cpu().numpy()[0][index]

                np.savetxt(fname=output_filename, X=np.asarray([data, data_label]).T)
                

            optimizer.step()
            optimizer.zero_grad()


            # adjust the w_rec to decouple the neurons
            adjust = False
            if adjust:
                with torch.no_grad():
                    net[1].w_rec *= mask_rec.to(device)

                    w_rec_block = torch.zeros((2,2), dtype=torch.float64).to(device=device)

                    for i in range(self.num_doa):
                        index_set = [2*i, 2*i+1]
                        w_rec_block += net[1].w_rec[index_set][:,index_set]
                    
                    w_rec_block /= self.num_doa
                    w_rec_block = torch.tensor(data=w_rec_block.detach().clone(), dtype=torch.float32)

                    for i in range(self.num_doa):
                        index_set = [2*i, 2*i+1]
                        net[1].w_rec[index_set][:,index_set] = w_rec_block


                print("w_rec block: ", w_rec_block.cpu().numpy())
            
                        
            if (epoch + 1) % 4 == 0: 
                plt.plot(prob[0].detach().cpu().numpy(), label="estimate")
                plt.plot(label.squeeze().detach().numpy(), label="original")
                plt.grid(True)
                plt.legend()
                plt.show(block=False)
                plt.pause(2)
                plt.close()
                
                # save the model
                model_path = os.path.join(Path(__file__).resolve().parent.parent, "models")
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                
                model_filename = os.path.join(model_path, f"model{time.asctime()}.pt".replace(" ", ""))
                torch.save(net.state_dict(), model_filename)

            



def test():
    # geometry
    num_mic = 7
    radius = 4.5
    speed = 340
    fs = 48_000

    geometry = CenterCircularArray(radius=radius, num_mic=num_mic, speed=speed)

    # spike encoder
    spike_encoder = ZeroCrossingSpikeEncoder(fs=fs, robust_width=8)

    # DoAs 
    num_doa = 4*num_mic + 1
    doa_list = np.linspace(-np.pi, np.pi, num_doa)

    # frequency range
    freq_range= [1_000, 2_000]

    # number of samples in the dataset
    size = 1000
    duration = 0.1

    dataset = SpikeDataset(
        geometry=geometry,
        spike_encoder=spike_encoder,
        doa_list=doa_list,
        freq_range=freq_range,
        size=size,
        duration=duration,
        fs=fs,
    )

    beamforming_vecs = np.random.randn(num_mic, num_doa) + 1j * np.random.randn(num_mic, num_doa)

    num_epochs = 10
    trainer = RSNNBeamforming(beamforming_vecs=beamforming_vecs, dataset=dataset)
    trainer.evolve(num_epochs=num_epochs)

test()