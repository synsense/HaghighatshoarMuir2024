# SNN-based audio source localization from a multi-microphone array, using the Hilbert transformation

This repository accompanies the paper "Low-power SNN-based audio source localisation using a Hilbert Transform spike encoding scheme" (https://arxiv.org/abs/2402.11748).

A Python implementation of a beamforming library is provided, as well as demos of live localisation using SNN inference, and scripts to generate all results figures in the paper.

<img src="figures/hilbert_beamforming/microphone_array.svg" alt="circular_microphone_array" style="zoom:200%;" />

# Brief introduction

Sound source localisation is used in many consumer electronics devices, to help isolate audio from individual speakers and to reject noise. Localization is frequently accomplished by "beamforming" algorithms, which combine microphone audio streams to improve received signal power from particular incident source directions.

These algorithms generally use knowledge of the frequency components of the audio source, along with the known microphone array geometry, to analytically phase-shift microphone streams before combining them. A dense set of band-pass filters is often used to obtain known-frequency "narrowband" components from wide-band audio streams.

<img src="figures/hilbert_beamforming/narrowband_beamforming.svg" alt="narrowband_beamforming" style="zoom: 150%;" />

Our novel method for sound source localisation from arbitrary microphone arrays is designed for efficient implementation using spiking neural networks (SNNs). We use a novel short-time Hilbert transform (STHT) to remove the need for demanding band-pass filtering of audio, and introduce a new accompanying method for audio encoding with spiking events.

We achieve state-of-the-art accuracy for SNN methods, comparable with traditional non-SNN super-resolution approaches.

## Installation

Install the `multimic` package. We recommend using a fresh `conda` or other python environment:

```bash
> git clone https://github.com/synsense/HaghighatshoarMuir2024.git
> cd HaghighatshoarMuir2024
> conda create --name HaghighatshoarMuir2024 python=3.8 cmake
...
> conda activate HaghighatshoarMuir2024
> pip install .
```

This will install the package and all required dependencies.

All packages and code have been tested with Python 3.8. `cmake` is required to build some dependencies.

## Generating the figures

The figure generation scripts are in subdirectory `/paper_plots` . Run all of the python files in turn to generate figures, **from the base directory**.

```bash
> cd paper_plots
> python SCRIPT_NAME.py
...
```

### Beam patterns for various methods
These scripts generate and plot beam patterns obtained from the various beamforming algorithms.

| Script                             | Method                                                       |
| ---------------------------------- | ------------------------------------------------------------ |
| `array_resolution_music.py`        | MUSIC beamforming                                            |
| `array_resolution_snn.py`          | SNN (float32) implementation of SNN Hilbert beamforming      |
| `array_resolution.py`              | Hilbert beamforming (non-SNN implementation)                 |
| `multiple_targets_beamformer.py`   | Conventional super-resolution beamforming, under multiple audio sources |
| `multiple_targets_music.py`        | MUSIC beamforming, under multiple audio sources              |
| `multiple_targets_snn.py`          | Hilbert SNN beamforming, under multiple audio sources        |
| `array_resolution_linear_music.py` | MUSIC beamforming on a linear microphone array               |
| `array_resolution_linear_snn.py`   | Hilbert SNN beamforming on a linear microphone array         |
| `array_resolution_random_music.py` | MUSIC beamforming on a random microphone array               |
| `array_resolution_random_snn.py`   | Hilbert SNN beamforming on a random microphone array         |

### Analysis of target localisation performance

These scripts analyse the performance of the various beamforming and DoA estimation approaches on noisy wideband and noisy speech signals.

| Script name                            | Method                                                       |
| -------------------------------------- | ------------------------------------------------------------ |
| `target_localization.py`               | Hilbert beamforming and DoA estimation (non-SNN implementation) |
| `target_localization_MUSIC.py`         | MUSIC algorithm                                              |
| `target_snn_localization.py`           | Hilbert beamforming with RZCC encoding and SNN inference for DoA estimation |
| `target_xylo_localization.py`          | Hilbert beamforming and DoA estimation implementation using the Xylo™ architecture, using bipolar RZCC spikes (+1, -1). |
| `target_xylo_unipolar_localization.py` | Hilbert beamforming and DoA estimation implementation using the Xylo™ architecture, using unipolar RZCC spikes (+1). |

### Simple visualisations

| Script name                            | Description                                                       |
| -------------------------------------- | ------------------------------------------------------------ |
| `phase_plot.py`                  | Visualise the phase of two overlapping complex exponential signals. |
| `random_phase.py`                | Visualise the phase of a random wideband signal, using the Hilbert transformation. |
| `rzcc_plots.py`                  | A simple visualisation of zero-crossing spike detection.     |
| `short_hilbert_transform.py`     | Visualise the frequency spectrum and performance of the STHT and RZCC encoding. |
| `chirp_phase_plot.py`            | Visualise the instantaneous phase of a chirp signal.         |

# Running demos

If you happen to have a microphone array development board connected to your PC, you can run a live demo of the localization algorithm. The demos assume you can record a seven-stream audio input. The scripts are all in the `micloc` directory:

```bash
> python micloc/SCRIPT_NAME.py
...
```



| Script name                  | Method                                                       |
| ---------------------------- | ------------------------------------------------------------ |
| `localization_demo.py`       | Hilbert beamforming and DoA estimation (non-SNN implementation) |
| `localization_demo_snn.py`   | Hilbert beamforming with RZCC encoding and SNN inference for DoA estimation |
| `localization_demo_MUSIC.py` | MUSIC algorithm                                              |
