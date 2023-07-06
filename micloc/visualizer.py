# -----------------------------------------------------------
# This module implements a simple online visualizer for SNN localization demo.
# 
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 06.07.2023
# -----------------------------------------------------------

import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Queue
import warnings

__all__ = ["Visualizer"]

# ===========================================================================
#                        Shared Queue used in Visualizer
# ===========================================================================
global_queue = Queue()


# ===========================================================================
#                            Visualizer Class
# ===========================================================================
class Visualizer:
    def __init__(self, buffer_size: int, dim_samples: int, waiting_time: float = 1.0):
        """this class builds a simple online visualizer.

        Args:
            buffer_size (int): number of samples to be shown/plotted.
            dim_sample (int): dimension of the samples
            waiting_time (int): how much the visualizer waits before prompting a message showing that it has not received any data to plot. Defaults to 1.0 sec.
        """
        self.buffer_size = buffer_size
        self.dim_samples = dim_samples

        self.buffer = np.zeros((self.buffer_size, self.dim_samples))
        self.time_vec = np.zeros(self.buffer_size)

        self.waiting_time = waiting_time

        self.active = False

        # start time of the simulation
        self.start_time = 0

    def start(self, **kwargs):
        """
        this function starts plotting the data.
        """
        # if there is any previous processes running kill it
        if hasattr(self, 'plotter'):
            warnings.warn(
                "an active plotter already exists! closing the plotter! make sure to call `stop` when a viusalization phase is over!"
            )

            if self.plotter.is_alive():
                # stop the plotter
                self.stop()

        # register the start time
        self.start_time = time.time()

        self.active = True
        self.time_vec = np.zeros(self.buffer_size)
        self.buffer = np.zeros((self.buffer_size, self.dim_samples))

        # block = False -> does not matter if some data points are lost
        global_queue.put((self.active, np.copy(self.time_vec), np.copy(self.buffer)), block=True)

        # arguments used for the figures
        fig_args = {
            "figsize": (16, 10),
            "xlabel": "",
            "ylabel": "",
            "title": "",
            "grid": False,
            "linewidth": 1,
            "linestyle": "-",
            "marker": ".",
            "label": "",
        }

        for key in kwargs.keys():
            fig_args[key] = kwargs[key]

        self.plotter = mp.Process(target=self.visualize, args=(fig_args, self.waiting_time))
        self.plotter.start()

    def stop(self):
        self.active = False

        # report it to the plotter: block=True to make sure that plotter sees this
        global_queue.put((self.active, np.copy(self.time_vec), np.copy(self.buffer)), block=True)

    def push(self, data_in: np.ndarray):
        """this function registers the input event in the buffer

        Args:
            data_in (np.ndarray): input data to be registered.
        """
        # register the time relative to the start time
        time_point = time.time() - self.start_time

        # put the data and time
        self.time_vec[:-1] = self.time_vec[1:]
        self.buffer[:-1, :] = self.buffer[1:, :]

        self.time_vec[-1] = time_point
        self.buffer[-1, :] = data_in

        # push the data into the queue
        # block = False -> does not matter if some data points are lost
        global_queue.put((self.active, np.copy(self.time_vec), np.copy(self.buffer)), block=True)

    @staticmethod
    def visualize(fig_args, waiting_time: float = 1.0):
        print("process started visualization ... ")

        # prepare the figure
        plt.figure(figsize=fig_args["figsize"])

        while True:
            # get the data from queue
            try:
                (active, time_vec, buffer) = global_queue.get(block=False)

                # set the last data input time
                last_data_time = time.time()
            except Exception as e:
                passed_time = time.time() - last_data_time

                if passed_time > waiting_time:
                    print(f"\n\nno data is transferred to visualizer in the last {passed_time}")
                    print("make sure to call `stop` function to finish visualization if there is no more data!")
                    print("waiting for 1 sec to receive the next data ....\n\n")
                    time.sleep(1)

            if not active:
                break

            plt.clf()
            plt.plot(time_vec, buffer, linewidth=fig_args["linewidth"], marker=fig_args["marker"], label=fig_args["label"])
            if fig_args["label"] != "":
                plt.legend()
            plt.xlabel(fig_args["xlabel"])
            plt.ylabel(fig_args["ylabel"])
            plt.title(fig_args["title"])
            plt.grid(fig_args["grid"])
            plt.draw()

            plt.show(block=False)
            plt.pause(0.01)

        print("end of visualization!")
        plt.close()


# ===========================================================================
#                            Some Test Cases
# ===========================================================================
def test_visualizer():
    # create a visualizer
    buffer_size = 20
    dim_samples = 2
    vz = Visualizer(buffer_size=buffer_size, dim_samples=dim_samples)

    # start the visualization
    vz.start(figsize=(12, 12), xlabel="time", ylabel="data", title="simple random data")

    num_samples = 2 * buffer_size
    period = 1

    print(f"start visualizer with {buffer_size} coming with period {period} sec")
    start_time = time.time()
    for _ in range(num_samples):
        data_in = np.random.randn(dim_samples)
        print(f"data: {data_in} - pushed at time: {time.time() - start_time}")
        vz.push(data_in=data_in)
        time.sleep(period)

    print("stop the visulaize for 3 sec")
    # vz.stop()

    time.sleep(5)

    print(f"start visualizer again with {buffer_size} coming with period {period} sec")
    vz.start()
    for _ in range(num_samples):
        data_in = np.random.randn(dim_samples)
        print(f"data: {data_in} - pushed at time: {time.time() - start_time}")
        vz.push(data_in=data_in)
        time.sleep(period)

    print("end of visualization!")
    vz.stop()


def main():
    test_visualizer()


if __name__ == '__main__':
    main()
