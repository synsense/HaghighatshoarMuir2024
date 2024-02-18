# ----------------------------------------------------------------------------------------------------------------------
# This module builds a simple audio recorder for multi-mic devkit.
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

import numpy as np
import time
import os
from pathlib import Path
import warnings


class MultiMicDevKitNotFound(Exception):
    pass


class AudioRecorder:
    def __init__(self):
        """this class reads audio devices and detects the DevKit so that the audio can be recorded."""

        # TODO: check all devices and see if devkit is among them
        pass

    def record_file(
        self, duration: float, bits: int = 16, fs: float = 48_000, report=False
    ):
        """this module records audio from miniDSP board and writes it to a file.

        Args:
            duration (float): duration of the signal to be recorded.
            bits (int): number of bits used for recording the input audio signal.
            fs (float): sampling rate.
            report (bool, optional): report various stages of recording files. Defaults to False.
        """
        import subprocess
        from scipy.io import wavfile

        filename = os.path.join(Path(__file__).parent.resolve(), ".temp_output.wav")

        # how much buffering we intend to have
        buffer_duration = 2 * duration

        # buffer size in bits: considering 8 channel microphones
        buffer_bit_size = int(fs * buffer_duration * 8)

        command = f'sox -b {bits} -e signed-integer -r {fs} -c 8 -d --clobber --buffer {buffer_bit_size} "{filename}" trim 0 {duration}'

        if report:
            print("+" * 100)
            print("start recording ....")
        start = time.time()
        output = subprocess.run(command, shell=True)

        if report:
            print(output.stdout)

        if output.returncode != 0:
            raise MultiMicDevKitNotFound(
                f"there was a problem with the board! recording failed!\n{output.stderr}"
            )

        duration_record = time.time() - start
        if report:
            print(f"\nfinished recording! duration: {duration_record}")

        # read the data from the temporary file
        rate, data = wavfile.read(filename)

        warnings.warn(
            f"The data format is np.intxx. For practical applications, it is better to convert it into np.floatxx to avoid overflow or underflow issues.\n"
            + "For example, in np.intxx format, an overflow may happen even in simple operations such as computing power `np.mean(data**2)` since\n"
            + "there is a very high chance that np.intxx square does not fit in np.intxx. This may cause weird errors!\n"
        )

        return np.asarray(data)
