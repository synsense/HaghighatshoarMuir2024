# Multi-Mic Localization
In this project, we use the data received from a microphone array of consisting of $M$ microphones to do localization and DoA (direction of arrival) estimation. The microphones are arranged in a circular geometry of radious $R$ and allow to detect the DoA of the target in the full angular range $\phi \in [-\pi, 
\pi]$. 

The goal of localization is to relative delay in the signals received from various microphones to detect the DoA of the incoming signal. 

# Signal Model
We denote the incomping signal from $M$ microphones by $X(t)=(x_1(t), \dots, x_M(t))^T$ where $x_i(t)$ denotes the time-domain signal received from microphone $i\in[M]$. Suppose we have an audio source at DoA $\phi$ that transmits the time-domain signal $x(t)$. Then the received signal at microphone $i$ is given by 
$$r_i(t)= x_i(t) + z_i(t) = x(t-\tau_i(\theta)) + z_i(t),$$
where $z_i(t)$ denotes the additive noise in microphone $i$ and where $\tau_i(\theta)$ denotes the delays from audio source to microphone $i \in [M]$ which depends on the DoA $\phi$. We always consider a far-field scenario where the audio source is far from the array such that the received signal is in the form of a wave with a flat wavefront such that we can write $\tau_i(\theta)$ as 
$$\tau_i(\theta) = \tau_0 - \frac{R \cos(\theta - \thete_i)}{c},$$
where $tau_0 = \frac{D}{c}$ with $D$ denoting the distance of the audio source from the center of the array, where $c \approx 340$ m/s denotes the speed of sound in the environment, where $R$ is the radiuos of the array, and where $\thete_i$ denotes the angle of the microphopne $i$ in the circular array configuration. 

<center>
<figure>
<img src="./figs/multi-mic.png" alt="multi-mic array" width="600">
<figcaption>Fig.1. A circular array consisting of several microphones receiving signal from an audio source. </figcaption>
</figure>
</center>


# Beamforming for DoA Estimation
There is a vast literature on how to recognize the DoA of the single target or perhaps multiple targets from the received signal $X(t)$. Perhaps the most well-known case is when the audio signal $x(t)$ is a narrowband sinusoidal signal of the form $x(t)=A \sin(2\pi f_0 t)$ with a frequency $f_0$. The wavelength of such a narrowband signal is given by $\lambda = \frac{c}{f_0}$. The precision of the estimated DoA $\phi$ depends on two factors:
- first the array elements should be spaced closely enough to so that the array elements provide sufficient spatial sampling. Rougly speaking, the spacing between the array elements should not be larger than $\frac{\lambda}{2}$ otherwise spatial aliasing will happen namely two different DoAs $\phi\neq \phi$ are mapped into the same input signal $X(t)$ such that they are not uniquely identifiable. This is known as the grating lope effect in the array processing literature.
- second to obtain the highest angular resolution, the spatial span of the array should be as large as possible.

In practical designs, one need to choose the array radius large enough to fulfill the resolution condition. And once the radious was fixed, one needs to choose enough number of microphones $M$ to make sure that their spacing is smaller than $\frac{\lambda}{2}$, i.e., 

$$\Delta = 2 R \sin(\frac{\pi}{M}) \approx \frac{2\pi R}{M} \leq \frac{\lambda}{2} = \frac{c}{2f_0}.$$

This relation can be also written as 

$$ f_0 \leq M \frac{c}{4 \pi R} =: f^*,$$
which implies that an array with fixed radius $R$ and number of microphones $M$ can only detect DoA $\phi$ unambigously only when the frequency of the incomping signal is less than $f^* = M \frac{c}{4 \pi R}$. 

In practice, the signal of the audio source $x(t)$ is not sinusoid and may have an arbitrary and unknown spectrum. The traditional way in those cases is to decompose the input signal into a collection of narrowband components via FFT-like methods

$$X(t) = \sum_ {f\in \mathcal{F}} X_f(t),$$

where $\mathcal{F} = \{f_1, f_2, \dots, f_F\}$ is a collection of $F$ frequencies to which the input signal is decomposed. One then applies narrowband beamforming at frequency $f \in \mathcal{F}$ to each component $X_f(t)$ and aggergates the results to estimate the DoAs of the target/targets.


# Novel Method of Beamforming Based on Hilbert Transform

## Brief introduction to Hilbert transform
Hilbert transform is well-known transform in signal processing application which is applied to recover the conjugate or phase-rotated vesrion of the input signal. Hilbert transfrom is a linear transformation and can be described but its impulse response $h(t) = \frac{1}{\pi t}$. There is  however some difficulty in the definition of the output due to singularity of the impulse response at $t=0$.

In signal processing applications, one can get a better picture of Hilbert transform in the frequency domain where it can be described by the frequency response $H(f) = -j \text{sign}(f)$ where $\text{sign}$ denotes the signum function. Intuitively speaking, Hilbert transform simply rotates the frequency components of the signal by $-\frac{\pi}{2}$. In particular, for a sinusoid signal $x(t)=\cos(2\pi f_0 t)$ it yields $\hat{x}(t) = \cos(2\pi f_0 t - \frac{\pi}{2}) = \sin(2 \pi f_0 t)$. 

Hilbert transform is particularly important when the input signal $x(t)$ is real-valued where in that case one can consider $x_a(t) = x(t) + j \hat{x}(t)$ as the analytic version of the signal. For example, in the case of harmonic signal 
$x(t)=\cos(2\pi f_0 t)$, we have that 

$$x_a(t) = \cos(2\pi f_0 t) + j \sin(2\pi f_0 t) = e^{j 2\pi f_0 t},$$

 which is called analytic since it has only positive frequency components as its Fourier transform is given by $\delta(f - f_0)$ which is non-zero only on the positive part of the spectrum.

 
## Application of Hilbert transform in Beamforming
Let $x(t)$ be a real-valued signal and let $x_a(t)= x(t) + j \hat{x}(t)$ be its corresponding analytic signal. We define the envelope and phase of the analytic signal as $e(t) = \sqrt{x(t)^2 + \hat{x}(t)^2}$ and $\phi(t) = \tan^{-1}(\frac{\hat{x}(t)}{x(t)})$ and write $x_a(t)$ as $x_a(t) = e(t) e^{j\phi(t)}$. Note that for well-defined functions $x(t)$, both $e(t)$ and especially $\phi(t)$ can be defined to be a continuous and well-defined function of $t$.

By applying the Fourier transform to $x_a(t) = x(t) + j \hat{x}(t)$, we can see that 

$$X_a(f) = X(f) \times (1 + j \times -j \text{sign}(f)) = X(f) \times (1 + \text{sign}(f)) = X(f) \times 2 u(f),$$

where it is seen that the analytic signal is zero for all negative frequencies $f<0$. 

One of the interesting implications of this result is that the phase $\phi(t)$ seems to be be an almost increasing function of $t$. Intuitively, this comes from the fact that, for inverse Fourier transform expression 

$$x_a(t) = \int_{-\infty}^{\infty} X_a(f) e^{j 2\pi f t} df = \int_{0}^{\infty} X_a(f) e^{j 2\pi f t} df,$$

we can write $x_a(t)$ as a super position of complex exponential expressions $e^{j 2\pi f t}$ with all positive frequencies $f$. So over all the phase of the signal should be almost an increasing function of $t$. 

This statement is of course not completely true as can be illustrated with a simple example of $e_1 e^{j \phi_1(t)} + e_2 e^{j \phi_2(t)}$ where both $\phi_1(t)$ and $\phi_2(t)$ are increasing functions of $t$. Assuming that $e_1 > e_2$ we can see that we can write this as
$$e_1 e^{j \phi_1(t)} \times (1 + \frac{e_2}{e_1} e^{j (\phi_2(t) - \phi_1(t)) }).$$

Since $\frac{e_2}{e_1} < 1$, one can show that the phase of the second term is just bounded within $[-\phi_{\max}, \phi_{\max}]$. As a result, the phase of the whole signal is given by $\phi_1(t) + b(t)$ where $b(t)$ is a bounded function with $|b(t)| \leq \phi_{\max}$. This simply implies that the final phase has an increasing part given by the phase of the strong one $\phi_1(t)$ plus a fluctuation part which has bounded amplitude so it should be almost increasing.

This simple example alos illustrates that in general phase may show a very complicated behavior by sligtly modifying the amplitudes, i.e., by making $e_1$ sligtly smaller than $e_2$, phase may switch from $\phi_1(t)$ to $\phi_2(t)$ after neglecting the bounded component. This extreme case of course does not happen when we have superposition of a large number of exponentials.

Now let us consider the audio signal $x(t)$ and its corresponding analytic signal $x_a(t) = e(t) e^{j \phi(t)}$. Since both the Hilbert transform and the propagation model are linear, when When $x(t)$ is transmitted by the audio source, the analytic version of the signal received in the microphone $i$ is given by 
$$x_a(t-\tau_i(\theta)) = e(t-\tau_i(\theta)) e^{j \phi(t-\tau_i(\theta))} \approx e(t) e^{j \phi(t-\tau_i(\theta))},$$
where we used the fact that the enevlope $e(t)$ varies slowly with time such that $e(t) \approx e(t-\tau_i(\theta))$. As a result, the major source of variation in the analytic signal is due to phase signal $\phi$. 

We will show that the relative variation in the phase signal across $M$ microphones can be used for beamforming. Note that the beamforming based on the Hilbert transform and phase of the analytic signal is more general than the narrowband version and yields the narrowband version when the signal is a sinusoid. More specifically, for $x(t) = A \cos(2 \pi f_0 t)$ the anlytic signal is given by $x_a(t) = A e^{j 2\pi f_0 t}$ which has a linear phase $\phi(t) = 2\pi f_0 t$ with the slope $2\pi f_0$. For such a narrowband signal, we can see that the effect of the delay in each array element is add the phase difference $2\pi f_0 \tau_i(\theta)$. It is conventional to pose this as array respones vector 
$$a_{f_0}(\theta) = (e^{j 2 \pi f_0 \tau_1(\theta)}, \dots, e^{j 2 \pi f_0 \tau_M(\theta)} ) ^T,$$
which encodes the phase variation at a given frequency $f$ as a function of DoA $\theta$ of the incoming signal. As a result, one can detect the DoA of the incoming signal as
$$\hat{\theta} = \text{argmax}_{\theta \in [-\pi, \pi]} \int |a_{f_0}(\theta)^H X_a(t)|^2 dt = \text{argmax}\ a_{f_0}(\theta)^H \Big ( \int X_a(t) X_a(t)^H dt \big ) a_{f_0}(\theta) = \text{argmax}\ a_{f_0}(\theta)^H  \hat{C}\ a_{f_0}(\theta)$$
where $\hat{X}(t)$ denotes the $M$-dim signal consisting of the analytic versions of the signals received in $M$ microphones and where we defined the time-covarince of this signal as the $M\times M$ matrix defined by
$$\hat{C} = \int X_a(t) X_a(t)^H dt.$$

One of the implications of this result is that we can pose the DoA estimation as the following step-by-step method:

- we choose specific angular precision and quantize the range of DoA into a grid $\Theta = \{\theta_1, \dots, \theta_G\}$ of size $G$ where $G = \alpha M$ with $\alpha$ deneotiong the oversampling factor. 
- we assume that the audio source produces the signal $x(t)$ and for each $\theta \in \Theta$ compute the signal received in the array.

- we apply the Hilbert transform and build the vector analytic signal $X_a(t)$ in the array.
- we compute the sample covariance matrix of $X_a(t)$ and obtain the $M\times M$ PSD maatrix $\hat{C} = \int X_a(t) X_a(t)^h dt$.
- we apply the SVD transform and compute the eigen-vector $u_\theta$ corresponding to $\hat{C}$, where we have used the supscipt $\theta$ to show that the singular vector corresponds to the DoA $\theta$. 
- we build the $M \times G$ beamforming matrix $U = [u_{\theta_1}, \dots, u_{\theta_G}]$.

We use the resulting beamforming matrix to detect the DoA of the incoming signal as follows:

- we receive the $M$-dim time-domain signal $X(t)$ in the array and apply Hilbert transform to each component to obtain the vector analytic signal $X_a(t)$.
- we apply the beamforming matrix $U$ and build the $G$-dim signal $U^H X_a(t)$.
- we accumulate the energy of the beamformed signal over an specific period of time of duartion $T$ to compute the average power over grid elements as a G-dim vector $$p = \frac{1}{T} \int_0^T |U^H X_a(t)|^2 dt.$$
- we find the DoA of the target/targets by identifying the large elements of $p$. For example, in the single target case, we have $\hat{\theta} = \text{argmax}_{\theta_i \in \Theta} p_i$.

# Generalization to discrete-time signals: limitations and approximations
In practical implementations, we have to use the sampled discrete-time version of the signal. An extension of Hilbert transform to discrete-time signals can be also developed. The main idea is to use the fact the Fourier transform of the analytic signal is given by $\hat{X}(f) = 2 X(f) u(f)$ and is obtained by truncating the negative components of the signal. By using this property we can define a similar Hilbert transform for discrete-time signals as follows.

Given a signal $x[n]$ of length $N$ we compute its $N$-point DFT $X[k]$ and then define the DFT of the analytic signal by dropping the negative frequency components which in the case of DFT transform corresponds to the second half of $X[k]$. There is a slight difference between odd and even values of $N$.

**$N$ is odd.** 
In such a case, we set $\hat{X}[0] = X[0]$ and $\hat{X}[k] = 2 X[k]$ for $k=1,\dots, \frac{N-1}{2}$ and set $\hat{X}[k]$ zero elsewhere.

**$N$ is even.** 
In such a case, we set $\hat{X}[k] = 2 X[k]$ for $k=0,1, \dots, \frac{N}{2}-1$ and set $\hat{X}[k]=0$ elsewhere.

A key property in both cases is that since the original signal is given by 

$$x[n] = \sum_{k} X[k] e^{j \frac{2\pi k}{N}n},$$

we are keeping those discrete harmonics that have a positive frequency $f_k = \frac{2k\pi}{N}$ such that the phase of each term $e^{j \frac{2\pi k}{N}n}$ is growing linearly with $n$. 

**How to define the phase?** In the case of continuous-time signals, the phase $\theta(t)$ of the anlytic signal is a contionuous function of $t$ so there is no ambiguity in defining phase. In the case of discrete-time signals, we need to define the unwrapped version of the phase signal, which satisfies $|\theta[n+1] - \theta[n]| < \pi$. In fact, given a phase signal whose components are all bounded in the interval $[-\pi, \pi]$ we can convert it into its unwrapped version by adding integer multiples of $2\pi$ to each term such that the condition $|\theta[n+1]-\theta[n]| < \pi$ is fulfilled. As in the continuous-time case, we may hope that the phase signal of an analytic signa is almost increasing since we have included only discrete exponentials $e^{j \frac{2k \pi}{N} n}$ whose phase is increasing with $n$ with a positive slope $\frac{2 k \pi}{N}$. 
 


