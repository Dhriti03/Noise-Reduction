# AI Noise Reduction 

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
*This is the original Audio File 
*![Original Audio file](https://user-images.githubusercontent.com/84375995/138569872-c4ef96a8-6d48-4f10-8ee7-c286de0a9c53.png)
*After Addition of the Noise 
*![Noise added to the signal](https://user-images.githubusercontent.com/84375995/138569964-aa06b69d-2b2b-4e92-af79-a8d265c120c7.png)
*The final Audio signal after removing noise 
*![The final Audio signal after removing noise ](https://user-images.githubusercontent.com/84375995/138569936-da5a10d8-1d3a-46c5-9c4d-aae682b1a583.png)


* On Manipulating the code according to your requirements, you could use it to control most of the Audio signlas .




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
## Acknowledgements and Resources

* [SRA VJTI](http://sra.vjti.info/) Eklavya 2021
* [Audacity](https://github.com/audacity/audacity/blob/master/src/effects) 
*[Noise Cancellation Method for Robust Speech 
Recognition](https://research.ijcaonline.org/volume45/number11/pxc3879438.pdf) 

<!-- LICENSE -->
## License

[License](LICENSE)
