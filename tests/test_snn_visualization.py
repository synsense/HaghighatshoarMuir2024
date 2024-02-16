# ----------------------------------------------------------------------------------------------------------------------
# This module builds a minimal visualizer for multi-mic board.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.07.2023
# ----------------------------------------------------------------------------------------------------------------------

from micloc.visualizer import Visualizer
from micloc.record import AudioRecorder
import numpy as np
import matplotlib.pyplot as plt


def test_simple():
    # build a microphone
    mic = AudioRecorder()

    # recording specs
    duration = 1
    fs = 48000
    time_vec = np.arange(0, duration, step=1 / fs)

    while True:
        audio = mic.record_file(
            duration=duration,
            fs=fs,
        )

        plt.figure()
        plt.plot(time_vec, audio)
        plt.xlabel("time (sec)")
        plt.ylabel("audio")
        plt.grid(True)
        plt.show(block=False)

        plt.pause(1)
        plt.close()


def test_snn_visualizer():
    # build a simple recorder
    mic = AudioRecorder()
    record_duration = 0.5
    num_bits = 32
    fs = 48_000

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
        ylabel="power of voice",
        title="power of voice signal received from microphone",
    )

    while True:
        # get the new data from microphones
        data = mic.record_file(
            duration=record_duration,
            bits=num_bits,
            fs=fs,
        )

        # compute the power of the signal
        print("input data dimension is: ", data.shape)
        power = np.mean(np.abs(data))
        print("received power from mics is: ", power)

        vz.push(power)


def main():
    # test_simple()
    test_snn_visualizer()


if __name__ == "__main__":
    main()
