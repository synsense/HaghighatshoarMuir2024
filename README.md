# multi-mic SNN localization

In this project, we restart the DoA estimation and localization project using a multi-mic array.

In this new version of the project, we have found a new method to resolve several issues:
- ambiguity in DoA estimation due to real-valued signals is resolved.
- a new method using online Hilbert trannsform is developed which has the potential to be used for a wide range of signal processing applications.


# Brief intro to beamforming
In the multi-mic localization project, we receive signals from M microphones. These signals are real-valued and we use 
Hilbert transform to make an analytic signal from them. The analytic signal is complex-valued and is used to design
beamforming matrices. 

One of the problems with Hilbert transform is that it requires having access to all the samples of the signal. In practice, however, we
would like to have an online version in which the input signal is processed regularly in short intervals and a result is computed.
These computed results then are assembled in time to decide on localization.

To do this, we convert the Hilbert transform into a windowed kernel version of specified duration $\Delta$. In this version,
if the input signal is of duration $\Delta$, the output of the kernel version would be the same as the original Hilbert transform, whereas
if the signal is longer, the kernel version will yield only an approximation of the Hilbert transform.

