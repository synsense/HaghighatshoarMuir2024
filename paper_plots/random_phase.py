# ----------------------------------------------------------------------------------------------------------------------
# This module produces the phase plot for a chirp signal and shows that it can be used to track the instantenous frequency.
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 28.08.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import hilbert, medfilt, lfilter, butter
from pathlib import Path
import os
from tqdm import tqdm

def use_latex():
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "xelatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

use_latex()


def plot_phase_random():
    # find the directory for this file
    root = os.path.join(Path(__file__).resolve().parent, "phase_random_plot")

    if not os.path.exists(root):
        os.mkdir(root)

    filename = os.path.join(root, "phase_random_plot.pgf")
    
    # chirp information
    fmin = 1_000
    fmax = 3_000
    num_period = 20
    duartion = num_period/fmin
    fs = 100 * fmax
    
    time_vec = np.arange(0, duartion, step=1/fs)
    num_sample = len(time_vec)
    # butterworth filter parameters
    N=4
    Wn = [fmin, fmax]
    b, a = butter(N, Wn, btype='pass', analog=False, output='ba', fs=fs)
    
    num_sim = 5
    plt.figure()
    
    for sim in range(num_sim):
        noise = np.random.randn(num_sample)
        sig_in = lfilter(b, a, noise)
        
        sig_h = sig_in + 1j*hilbert(sig_in)
        phase_est = np.unwrap(np.angle(sig_h))
        
        slope = (phase_est[-1] - phase_est[0])/duartion / (2*np.pi)
        
        plt.plot(time_vec, phase_est, label=f"signal {sim}: slope:{int(slope)}")
        
    plt.title(f"random noise with flat spectrum in [{fmin/1000}, {fmax/1000}] KHz")
    plt.legend()
    plt.grid(True)
    plt.xlabel("time")
    plt.ylabel("phase of HT")
    # plt.show()

    plt.savefig(filename)


def main():
    plot_phase_random()


if __name__ == '__main__':
    main()
