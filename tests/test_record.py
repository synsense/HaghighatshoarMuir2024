# ----------------------------------------------------------------------------------------------------------------------
# This module tests recording from the microphone.
#
#
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 06.07.2023
# ----------------------------------------------------------------------------------------------------------------------
from micloc.record import AudioRecorder
import numpy as np
import matplotlib.pyplot as plt


def test_record():
    mic = AudioRecorder()

    duration = 3
    fs = 48000

    audio = mic.record_file(
        duration=duration,
        fs=fs,
    )

    print("dimension of the recorded audio: ", audio.shape)
    print("data-type of the recorded audio: ", audio.dtype)

    # convert audio into float format to avoid issues
    audio = np.asarray(audio, dtype=np.float64)
    power_per_channel = np.mean(audio**2, axis=0)

    print("power per channels: ", power_per_channel)

    num_sample, dim = audio.shape
    time_vec = np.arange(num_sample) / fs

    plt.plot(time_vec, audio)
    plt.xlabel("time (sec)")
    plt.ylabel("audio")
    plt.grid(True)
    plt.show()


def main():
    test_record()


if __name__ == "__main__":
    main()
