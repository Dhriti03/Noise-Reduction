# Noise reduction using spectral gating in python
## Steps of algorithm
1. An FFT is calculated over the noise audio clip
2. Statistics are calculated over FFT of the the noise (in frequency)
3. A threshold is calculated based upon the statistics of the noise   (and the desired sensitivity of the algorithm)
4. An FFT is calculated over the signal
5. A mask is determined by comparing the signal FFT to the threshold
6. The mask is smoothed with a filter over frequency and time
7. The mask is appled to the FFT of the signal, and is inverted

## Load Data (Import Libraries )
~~~python
import IPython 
from scipy.io import wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
import wave
%matplotlib inline
~~~
1. Here we are importing the libraries like the **IPython** lib used for the  to create a comprehensive environment for interactive and exploratory computing.
2. From the **Scipy.io** library is used for manipulating the data and visualization of the data using a wide range of python commands .
3. **NumPy** contains a multi-dimensional array and matrix data structures. It can be utilised to perform a number of mathematical operations on arrays such as trigonometric, statistical, and algebraic routines thus is a very useful library .
3. **Matplotlib.pyplot** library  helps to understand the huge amount of data through different visualisations.
4. **Librosa** used when we work with audio data like in music generation(using LSTM's), Automatic Speech Recognition. It provides the building blocks necessary to create the music information retrieval systems.
5. **%matplotlib inline** to enable the inline plotting, where the plots/graphs will be displayed just below the cell where your plotting commands are written. It provides interactivity with the backend in the frontends like the jupyter notebook.


## Wave file 

~~~python
wav_loc = r'/home/Noise_Reduction/Downloads/wave/file.wav'
rate, data = wavfile.read(wav_loc,mmap=False)
~~~
Here we take the waw file path location and then read that waw file with the **wavefile** module which is from the **Scipy.io**  library.
with parameters (filename - string or open file handle which is a 
Input WAV file.) then the (mmap : bool, optional in which whether to read data as memory-mapped (default: False).

~~~python
def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real

~~~
1. Here we firstly define the fft noise function in brief , a Fast Fourier transform (FFT) is an algorithm that computes the discrete Fourier transform (DFT) of a sequence, or its inverse (IDFT). Fourier analysis converts a signal from its original domain (often time or space) to a representation in the frequency domain and vice versa. The DFT is obtained by decomposing a sequence of values into components of different frequencies.
2. Using fast fourier transform and defining a function of data type complex and finally calculating the real part of the function.
In this the freqencies ranging between minimum frequency and max frequency are set to 1 and rest unwanted are neglected.
3.  Giving the file location
4. Reading the wav file
5. -32767 to +32767 is proper audio (to be symmetrical) and 32768 means that the audio clipped at that point
6. wav-file is 16 bit integer, the range is [-32768, 32767], thus dividing by 32768 (2^15) will give the proper twos-complement range of [-1, 1]

~~~python
def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1): 
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate)) 
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)
~~~
1. A function or time series whose Fourier transform is restricted to a finite range of frequencies or wavelengths.
2. defining the freq with the standard freq with the min and max limit.

~~~python
IPython.display.Audio(data=data, rate=rate)
~~~

1. Create an audio object.
2. When this object is returned by an input cell or passed to the display function, it will result in Audio controls being displayed in the frontend.

~~~python
ploting the given data 
fig, ax = plt.subplots(figsize=(20,4))
ax.plot(data)
~~~
    
1. Here we plot the frequency curve wrt the information on the x axis and the data of noise provided.
2. **plt.subplots()** is a function that returns a tuple containing a figure and axes object(s). Thus when using fig, **ax = plt.subplots()** you unpack this tuple into the variables fig and ax. Having fig is useful if you want to change figure-level attributes or save the figure as an image file later.

~~~python
noise_len = 2 # seconds
noise = band_limited_noise(min_freq=4000, max_freq = 12000, samples=len(data), samplerate=rate)*10
noise_clip = noise[:rate*noise_len]
audio_clip_band_limited = data+noise
~~~


1. The Band-Limited White Noise block specifies a two-sided spectrum, where the units are Hz.
2. where the max of 12000 and min freq of 4000 is compared wrt the noise and the data provided.
3. here we are clipping the noise signal by having a product of rate and the len of the noise signal.
4. thus adding the noise and the given data 
4. In effect, adding noise expands the size of the training dataset. 
5. random noise is added to the input variables making them different every time it is exposed to the model.
6. Adding noise to input samples is a simple form of data augmentation.
7. Adding noise means that the network is less able to memorize training samples because they are changing all of the time,
8. resulting in smaller network weights and a more robust network that has lower generalization error.

~~~python 
fig, ax = plt.subplots(figsize=(20,4))
ax.plot(audio_clip_band_limited)
IPython.display.Audio(data=audio_clip_band_limited, rate=rate)
~~~
1. Here we are plotting that  graph of the audio signal which is clipped the data with the noise being added .
2. Then display that audion signal with **IPython.display.Audio**.

~~~python 
import time
from datetime import timedelta as td
~~~
1. **import time** This module provides various time-related functions. For related functionality, see also the datetime and calendar modules.
class datetime.timedelta
2. A duration expressing the difference between two date, time, or datetime instances to microsecond resolution.
~~~python 
def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
~~~
1. Short Time Fourier Transform can be used to quantify change of a nonstationary signal’s frequency and phase content over time.
2. Hop length should refer to the number of samples in between successive frames. For signal analysis, Hop length should be less than the frame size, so that frames overlap.
3. Parameters
ynp.ndarray [shape=(n,)], real-valued
input signal

~~~python
n_fftint > 0 [scalar]
~~~
length of the windowed signal after padding with zeros. The number of rows in the ***STFT matrix D is (1 + n_fft/2)***. The default value, n_fft=2048 samples, corresponds to a physical duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the default sample rate in librosa. This value is well adapted for music signals. However, in speech processing, the recommended value is 512, corresponding to 23 milliseconds at a sample rate of 22050 Hz. In any case, we recommend setting n_fft to a power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm.

~~~python
hop_lengthint > 0 [scalar]
~~~
number of audio samples between adjacent STFT columns.

Smaller values increase the number of columns in D without affecting the frequency resolution of the STFT.

If unspecified, defaults to win_length // 4 (see below).

~~~python 
win_lengthint <= n_fft [scalar]
~~~
Each frame of audio is windowed by window of length win_length and then padded with zeros to match **n_fft**.

Smaller values improve the temporal resolution of the STFT (i.e. the ability to discriminate impulses that are closely spaced in time) at the expense of frequency resolution (i.e. the ability to discriminate pure tones that are closely spaced in frequency). This effect is known as the time-frequency localization trade-off and needs to be adjusted according to the properties of the input signal y.

If unspecified, defaults to **win_length = n_fft** .

~~~python def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)
~~~
1. Inverse short-time Fourier transform (ISTFT).Converts complex-valued spectrogram stft_matrix to time-series y by minimizing the mean squared error between stft_matrix and STFT of y as described in 
2. In general, window function, hop length and other parameters should be same as in stft, which mostly leads to perfect reconstruction of a signal from unmodified stft_matrix.

~~~python 
def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)
~~~
1.Convert an amplitude spectrogram to dB-scaled spectrogram.This is equivalent to power_to_db(S**2), but is provided for convenience.

~~~python def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)
~~~
1. Convert a dB-scaled spectrogram to an amplitude spectrogram.
2. This effectively inverts amplitude_to_db:
3. **db_to_amplitude(S_db) ~= 10.0**(0.5 * (S_db + log10(ref)/10))**

~~~python
def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(                
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
~~~
1. Ploting the spectogram with signal as the input .
2. Axes Class contains most of the figure elements: Axis, Tick, Line2D, Text, Polygon, etc., and sets the coordinate system.
3. It provides multiple colour maps in matplotlib accessible  via this function .o find a good representation in 3D colorspace for your data set. 
~~~python
fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
~~~
1. The best way to see what's happening, is to add a colorbar (plt.colorbar(), after creating the scatter plot). You'll note that your out values between 0 and 10000 are all below the lowest part of the bar, where things are a very light green.

2. In general, values below vmin will be colored with the lowest color, and values above vmax will get the highest color.
3. If you set vmax smaller than vmin, internally they will be swapped. Although, depending on the exact version of matplotlib and the precise functions called, matplotlib might give an error warning. So, best to set vmin always lower than vmax.

~~~python 
def plot_statistics_and_filter(            
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise") 
                                                                       
                                                                       
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")  
    ax[0].set_title("Threshold for mask")    
                                             
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask") 
                                                
    plt.show()
~~~
1.  Plots basic statistics of noise reduction.
2. Signal-to-noise ratio (SNR or S/N) is a measure used in science and engineering that compares the level of a desired signal to the level of background noise. 
3. SNR is defined as the ratio of signal power to the noise power, often expressed in decibels. 
4. A ratio higher than 1:1 (greater than 0 dB) indicates more signal than noise.
5. Setting up the threshhold frequency for noise masking.
6. Masking threshold refers to a process where one sound is rendered inaudible because of the presence of another sound.
7. So the masking threshold is the sound pressure level of a sound needed to make the sound audible in the presence of another noise called a "masker"
8. thus added the threshold .
9. Blur noise signals with various low pass filters
10. Apply custom-made filters to images (2D convolution)

~~~python
def removeNoise(   # to average the signal (voltage) of the positive-slope portion (rise) of a triangle wave to try to remove as much noise as possible. 

    audio_clip,    # these clips are the parameters used on which we would do the respective operations 
    noise_clip,
    n_grad_freq=2,    # how many frequency channels to smooth over with the mask.
    n_grad_time=4,    # how many time channels to smooth over with the mask.
    n_fft=2048,       # number audio of frames between STFT columns.
    win_length=2048,  # Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
    hop_length=512,   # number audio of frames between STFT columns.
    n_std_thresh=1.5, # how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
    prop_decrease=1.0, #To what extent should you decrease noise (1 = all, 0 = none)
    verbose=False,     # flag allows you to write regular expressions that look presentable
    visual=False,      #Whether to plot the steps of the algorithm
):
~~~
1. **def removeNoise(** 
 to average the signal (voltage) of the positive-slope portion (rise) of a triangle wave to try to remove as much noise as possible.

2. **audio_clip,**   
 these clips are the parameters used on which we would do the respective operations 
3. **noise_clip,**
   **n_grad_freq=2**
 how many frequency channels to smooth over with the mask.
4. **n_grad_time=4,** 
how many time channels to smooth over with the mask.
5. **n_fft=2048**       
 number audio of frames between STFT columns.
6. **win_length=2048,**
 Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
7. **hop_length=512,**
 number audio of frames between STFT columns.
8. **n_std_thresh=1.5** 
 how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
10. **prop_decrease=1.0,** 
To what extent should you decrease noise (1 = all, 0 = none)
11. **verbose=False,**     
 flag allows you to write regular expressions that look presentable
**visual=False,**      #Whether to plot the steps of the algorithm
):
~~~python
if verbose:
        start = time.time() 
~~~
1. Time module in Python provides various time-related functions. This module comes under Python’s standard utility modules.

    **time.time()** method of Time module is used to get the time in seconds since epoch. The handling of leap seconds is platform dependent.
~~~python

    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  
~~~
1. STFT over noise
1. convert to dB

~~~python
 
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
~~~
1. Calculate statistics over noise
2. Here we for the thresh noise we add the mean and the standard noise and the n_std noise .

~~~python

    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
~~~
1. STFT over signal
~~~python

    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
~~~
1.  Calculate value to mask dB to

~~~python

    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
~~~
1. Create a smoothing filter for the mask in time and frequency
~~~python 

    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
~~~
1. calculate the threshold for each frequency/time bin

~~~python 

    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
 ~~~
 ~~~python    
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
~~~
~~~python     
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
~~~   
~~~python     
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
~~~  
~~~python      
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))        
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
~~~
~~~python        
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram")
    return recovered_signal

~~~
1.  mask if the signal is above the threshold
2. convolve the mask with a smoothing filter

# Thus the Noise reduction in python  is complete .







