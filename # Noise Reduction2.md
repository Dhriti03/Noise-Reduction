#  Noise Reduction 

<p align="center">
    
![image](https://user-images.githubusercontent.com/84375995/138570711-b802d8e7-f4c7-43aa-8f8f-da77cbbcd01d.png)


</p>  

<!-- TABLE OF CONTENTS -->

## Table of Contents

* [About the Project](#about-the-project)
  * [Tech Stack](#tech-stack)
  * [File Structure](#file-structure)
* [Getting Started](#getting-started)
* [Results and Demo](#results-and-demo)
* [Future Work](#future-work)
* [Contributors](#contributors)
* [Acknowledgements and Resources](#acknowledgements-and-resources)
* [License](#license)

<!-- ABOUT THE PROJECT -->

## About The Project
### Real time noise cancellation of the noise inthe Audio data signal .
*  The noise needed to be removed  which is naturally induced like the non environmental noise  which is removed with the denoising the signal  .
Refer this [documentation](https://github.com/Dhriti03/ai-noise-reduction/blob/master/AI%20Noise%20Reduction.pdf)
also this [Blog on AI noise reduction](https://github.com/Dhriti03/ai-noise-reduction/blob/master/Blog.pdf)

<hr>

### Tech Stack

* The [Librosa](https://github.com/librosa/librosa) Library for Audio manupulation is used.
* For the Audio signals we used [scipy](https://github.com/scipy/scipy) 
* [Matplotlib](https://github.com/matplotlib/matplotlib) used to manipulate the data and visualize the signal .
<!-- This section should list the technologies you used for this project. Leave any add-ons/plugins for the prerequisite section. Here are a few examples. -->

The rest is Numpy  for mathematical operations , wave for the operating on the wave file .

### File Structure

```
AI Noise Reduction 
├───docs                                                                       ## Documents and Images
│   └───Input Audio file
├───
Project Details 
│   |
│   ├───
│   │   ├───Research papers
│   │   ├───Linear Algebra 
│   │   ├───Neural networks & Deep Learning 
│   │   ├───Project Documentation
│   │   ├───AI Noise Reduction Blog
│   │   ├───AI Noise Reduction Report
│   │   └───Code Implementation
│   │       ├───AI Noise Reduction.py
│   │       ├───audio.wav
│   │       ├───Resources
```


# When some annoying noise springs up and you have put on your headphones
![image](https://github.com/Dhriti03/ai-noise-reduction/blob/dev/imp.png)
### That are cheap and don't offer Active Noise Cancellation , how do you feel ?

Definitely unpleasant. And the problem
is that you can’t take any action
because you don’t know when that
annoying sound will reappear.Noise. 
It’s all around us. Whether it’s a crying baby,
the gossiping of colleagues or even the
whistling of a pressure cooker. A quiet
place to work or enjoy some relaxation is becoming a truly premium commodity. Noise
cancellation technology can help bring the zen back into your life. Automation of
reducing audio noise is necessary even though we don’t have costly headphones. As
engineers we are bound to do something about it.

## A Basic overview of SOUND :
Sound is energy travelling as a wave through some form of matter. We generally hear
sound transmitted through the medium which is air.
For example, a guitar string vibrates in air, knocking air molecules about. Those
molecules knock into their neighboring molecules, and so on.We are known to terms
like Compression and Rarefaction. When the wave moving through the air hits our
eardrums, it moves them at the same frequency as the guitar string

Our brain converts that movement into electrical signals that you then perceive as
sound. That’s a very simplistic explanation of what sound is, but it’s the least you need
to know to understand noise cancellation.
The algorithm is used to calculate the anti-noise wave needed to destructively interfere
with the ambient noise where you are. Then a device known as a transducer, which is
essentially a type of speaker, generates the anti-noise wave

The end effect is that when you switch on the noise cancellation function, you’re
suddenly enveloped in (near) silence. Which allows you to enjoy your audio at much
better fidelity or just enjoy some peace and quiet.
I know you are wondering that the name of Article AI Noise Cancellation ,then where is
AI .

![image](https://github.com/Dhriti03/ai-noise-reduction/blob/dev/download%20(1).jpeg) 

## AI-Powered Noise Removal
A fairly new development is the application of artificial intelligence technology to
remove unwanted noise from an audio signal. Unfortunately this can’t quite happen in
real time just yet, so it isn’t effective at noise cancellation. What it is very good at is
removing noise from a microphone signal.
For example, if you’re trying to Skype someone or make a voice recording in a noisy
environment, it can be hard for the person on the other end to understand what you’re
saying. Using artificial intelligence, that audio stream can be analyzed and everything
but your own voice can be removed.
In this Part I of our series we discuss methods of how it’s done formally.

## Introduction
To implement this Algorithm we will require :
1. A signal audio clip containing the signal and the noise intended to be removed.
2. Knack to learn some concepts related to signal processing.
Process:
3. We have a signal that we can extract some information from.We can extract data
regarding amplitude,frequency,etc of the signal to convert it into the required
range.

### But wait! Notice how the exponential in the second term in the sum for z2 is the same?

1. As the exponential in the third term in the sum for z1. 
They are both equal to
exp⁡ (−2πi⋅1⋅2/n). There is no need to compute this exponential quantity twice.
2.  We can simply compute it the first time, store it in memory, and then retrieve it when it is needed
to compute z2(assuming that retrieving from memory is faster than computing it from scratch.) 
3. One can think of the FFT algorithm as an elaborate bookkeeping algorithm that keeps track
of these symmetries in computing the Fourier coefficients.
4. Suppose we have a series y1,y2,....,yn and we want to compute the complex Fourier
coefficient z1. 
5. Going by the formula in the previous section, this would require
computing.
![noise](https://github.com/Dhriti03/ai-noise-reduction/blob/dev/img3.png)

 ## Now ,the key function of the project (subtract/reduce noise):
### Algorithm to separate / subtract / reduce noise
1. ● After adding the noise signal ,it is to be removed to achieve the result.
2. ● To separate/subtract ,we are defining a threshold value
3. ● And then a remove function is used for masking,filtering ,etc.



What else we can do is : We can improve if the model learns by itself and
adapts the various surroundings and removes the various noises, may it be
environmental ,industrial or any. But it is quite more than the scope of this
blog.Hope,we will discuss it later ;)

## Now,in Part || ,we will go sequentially for further understanding .

The basic idea of the FFT is to apply divide and conquer. We divide the coefficient vector
of the polynomial into two vectors, recursively compute the DFT(Discrete Fourier
Transform) for each of them, and combine the results to compute the DFT of the
complete polynomial. Simple meaning is to reduce the complexity of the function we
use FFT.Fourier analysis converts a signal from its original domain (often time or space)
to a representation in the frequency domain and vice versa.

Here,we are creating a function having a parameter,which is to be assigned the array of
data type complexes. Also defining phases using random functions in python.Then after
converting phases into complex form like a+jb ,the real part is returned.
Now,we are setting some minimum and maximum value of frequency, sample and
sample rate to get the generated noise in the required frequency range .

Then by setting
frequency range, we will add this generated noise into our original data signal to output
7a signal containing noise which we have to remove in further implementation.Also
convert noise in dB.
To denoise the signal we are using a remove function having various parameters having
some conditions to check for threshold ,application,recovery,etc.

It masks(process where one sound is rendered inaudible because of the presence of another sound) if
the signal is above the threshold ,convolve the mask if with a smoothing filter.

We can also plot spectrograms of these processes. There is method time.time to calculate the required time by subtracting the initial one.After returning from the remove function ,the
noise will be reduced.

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites & Installation

* Tested on Windows

```sh
git clone https://github.com/Dhriti03/ai-noise-reduction.git
cd ai-noise-reduction
```

In your Notebook install certain libraries 

```cmd
pip install wave 
pip install librosa
pip install scipy.io
pip install matplotlib.pyplot

```

<!-- RESULTS AND DEMO -->
## Results and Demo
## Video clip of the project .
[![image](https://user-images.githubusercontent.com/84375995/138935212-453b9ace-65b0-4222-b0da-63e4c51dc2e1.png)](https://www.youtube.com/watch?v=9dDNMoTWkTw)

*This is the original Audio File 
*![Original Audio file](https://user-images.githubusercontent.com/84375995/138569872-c4ef96a8-6d48-4f10-8ee7-c286de0a9c53.png)
*After Addition of the Noise 
*![Noise added to the signal](https://user-images.githubusercontent.com/84375995/138569964-aa06b69d-2b2b-4e92-af79-a8d265c120c7.png)
*The final Audio signal after removing noise 
*![The final Audio signal after removing noise ](https://user-images.githubusercontent.com/84375995/138569936-da5a10d8-1d3a-46c5-9c4d-aae682b1a583.png)
*Flowchart for the project 
*![image](https://github.com/Dhriti03/ai-noise-reduction/blob/dev/downcccccload.png)

* On Manipulating the code according to your requirements, you could use it to control most of the Audio signlas .
##Theory 
# Noise reduction using spectral gating in python
## Steps of algorithm
1. An FFT is calculated over the noise audio clip
2. Statistics are calculated over FFT of the the noise (in frequency)
3. A threshold is calculated based upon the statistics of the noise   (and the desired sensitivity of the algorithm)
4. A mask is determined by comparing the signal FFT to the threshold
5. The mask is smoothed with a filter over frequency and time
6. The mask is appled to the FFT of the signal, and is inverted

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

    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  
~~~
1. STFT over noise
1. convert to dB

~~~python
 
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    
~~~
1. Calculate statistics over noise
2. Here we for the thresh noise we add the mean and the standard noise and the n_std noise .

~~~python

   
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    
~~~
1. STFT over signal
~~~python

    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    
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
    
 ~~~
 1. mask for the signal
 ~~~python    
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
~~~
1. Mask Convolution with Smoothning filter
  
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
~~~  
1. Mask the signal 
~~~python      
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
   
~~~
1. recover the signal
1.  Thus apply mask if the signal is above the threshold
2. convolve the mask with a smoothing filter

## Result :
The key idea behind the whole Algorithm is to create a signal which is out of phase
with the Noise signal.

Now,the AI Noise reduction algorithm is ready to be
implemented with the coding practice.Enjoy the uninterrupted conversations
calmly.

![image](https://github.com/Dhriti03/ai-noise-reduction/blob/dev/img4.png)
<!-- FUTURE WORK -->
## Future Work

- [x] Applying the Noise reduction algorithum for the already downloaded wav file. 
- [x] Applying the FFT over the live recording of the audio signal .
- [ ] Further more deep implementation of the AI for the Noise cancellation.
- [ ] Applying the Noise reduction Algorithum for various formats of Audio files .
- [ ] The live audio signal with the microphone and Esp32 and thus will get the wav file for the further computation and signal processing .

<!-- CONTRIBUTORS -->
## Contributors

* [Dhriti Mabian](https://github.com/Dhriti03)
* [Priyal Awankar](https://github.com/Pixels123priyal)


<!-- ACKNOWLEDGEMENTS AND REFERENCES -->
## Acknowledgements 

*[SRA VJTI_Eklavya 2021](https://sravjti.in/) 

# Mentors 
* [Shreyas Atre ](https://github.com/SAtacker)
* [Harsh Shah ](https://github.com/HarshShah03325)

 ## Resources
* [Audacity](https://github.com/audacity/audacity/blob/master/src/effects) 



<!-- LICENSE -->
## License

[License](LICENSE)
