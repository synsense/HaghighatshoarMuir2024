# ----------------------------------------------------------------------------------------------------------------------
# This module tests the designed multi-thread visualizer.
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 05.07.2023
# ----------------------------------------------------------------------------------------------------------------------

from micloc.visualizer import Visualizer
import numpy as np
import time


def test_visualizer():
    # build a simple visualizer
    buffer_size = 60
    dim_samples = 1
    vz = Visualizer(
        buffer_size=buffer_size,
        dim_samples=dim_samples,
        waiting_time=2,
    )

    vz.start(figsize=(16, 10), xlabel="time", ylabel="power of voice",
             title="power of voice signal received from microphone",
             grid=True,
             label="just a simple random number",
             )

    while True:
        # generate a random direction
        num_direction = 16
        direction = int(num_direction * np.random.rand(1)[0])

        vz.push(direction)

        # wait for 1 sec
        time.sleep(1)


def main():
    test_visualizer()


if __name__ == '__main__':
    main()
