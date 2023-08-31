# ----------------------------------------------------------------------------------------------------------------------
# This module produces the phase plot of two complex exponential to show that the phase may be non-monotone.
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

def plot_phase():
    # find the directory for this file
    root = os.path.join(Path(__file__).resolve().parent, "two_exp_phase_plots")

    if not os.path.exists(root):
        os.mkdir(root)

    filename = os.path.join(root, "two_exp_phase.pgf")

    # parameters of the system
    time_vec = np.linspace(0, 4*np.pi, 10000)

    amp_vec = [1, 0.9]
    slope_vec = [1, 2]

    sig_in = 0 
    for (amp, slope) in zip(amp_vec, slope_vec):
        sig_in = sig_in + amp * np.exp(1j * slope * time_vec)

    phase = np.unwrap(np.angle(sig_in))

    plt.figure()

    plt.subplot(211)
    plt.plot(time_vec,phase, linewidth=2, label=r"$\phi(t)$")
    plt.plot(time_vec, slope_vec[0] * time_vec, "--", linewidth=2, label=r"$\phi_1(t)$")
    plt.legend()
    plt.title(r"sum of two complex-exp: $e_1=1, e_2=0.9, \phi_1(t)=t, \phi_2(t)=2t$")
    plt.ylabel("phase")
    plt.grid(True)

    plt.subplot(212)
    plt.plot(time_vec, slope_vec[0] * time_vec, "--", linewidth=2, label=r"$ \phi_1(t)$")
    plt.plot(time_vec, slope_vec[1] * time_vec, "-.", linewidth=2, label=r"$ \phi_2(t)$")
    plt.plot(time_vec, phase - slope_vec[0] * time_vec, linewidth=2, label=r"$b(t)$")
    plt.ylabel("phase")
    plt.xlabel("time (s)")
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)


def main():
    plot_phase()


if __name__ == '__main__':
    main()
