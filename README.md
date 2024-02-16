# SNN-based audio source localization from a multi-microphone array, using the Hilbert transformation

This repository accompanies the paper "Low-power SNN-based audio source localisation using a Hilbert Transform spike encoding scheme" (DOI XXX).

A Python implementation of a beamforming library is provided, as well as demos of live localisation using SNN inference, and scripts to generate all results figures in the paper.

<img src="figures/hilbert_beamforming/microphone_array.svg" alt="circular_microphone_array" style="zoom:200%;" />

# Brief introduction

Sound source localisation is used in many consumer electronics devices, to help isolate audio from individual speakers and to reject noise. Localization is frequently accomplished by "beamforming" algorithms, which combine microphone audio streams to improve received signal power from particular incident source directions.

These algorithms generally use knowledge of the frequency components of the audio source, along with the known microphone array geometry, to analytically phase-shift microphone streams before combining them. A dense set of band-pass filters is often used to obtain known-frequency "narrowband" components from wide-band audio streams.

<img src="figures/hilbert_beamforming/narrowband_beamforming.svg" alt="narrowband_beamforming" style="zoom: 150%;" />

Our novel method for sound source localisation from arbitrary microphone arrays is designed for efficient implementation using spiking neural networks (SNNs). We use a novel short-time Hilbert transform (STHT) to remove the need for demanding band-pass filtering of audio, and introduce a new accompanying method for audio encoding with spiking events.

We achieve state-of-the-art accuracy for SNN methods, comparable with traditional non-SNN super-resolution approaches.

## Installation

* Install the `multimic` package. We recommend using a fresh `conda` or other python environment
  `> pip install multimic`
  This will install the package and all required dependencies.

## Generating the figures

The figure generation scripts are in subdirectory `/paper_plots` . Run all of the python files in turn to generate figures.

```bash
> cd paper_plots
> python SCRIPT_NAME.py
...
```

### Beam patterns for various methods
These scripts generate and plot beam patterns obtained from the various beamforming algorithms.

| Script                           | Method                                                       |
| -------------------------------- | ------------------------------------------------------------ |
| `array_resolution_music.py`      | MUSIC beamforming                                            |
| `array_resolution_snn.py`        | SNN (float32) implementation of SNN Hilbert beamforming      |
| `array_resolution.py`            | Hilbert beamforming (non-SNN implementation)                 |
| `multiple_targets_beamformer.py` | Conventional super-resolution beamforming, under multiple audio sources |
| `multiple_targets_music.py`      | MUSIC beamforming, under multiple audio sources              |
| `multiple_targets_snn.py`        | Hilbert SNN beamforming, under multiple audio sources        |

### Analysis of target localisation performance

These scripts analyse the performance of the various beamforming and DoA estimation approaches on noisy wideband and noisy speech signals.

| Script name                            | Method                                                       |
| -------------------------------------- | ------------------------------------------------------------ |
| `target_localization_MUSIC.py`         | MUSIC algorithm                                              |
| `target_localization.py`               | Hilbert beamforming and DoA estimation (non-SNN implementation) |
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

