Task: Identify three short-term spectral features (e.g., zero-crossing rate, spectral centroid) and three long-term spectral features (e.g., pitch, harmonicity) for audio input, explaining their significance in detecting deep fakes. List some libraries or techniques in python that would allow for such features to be numerically represented.

## Spectral Features: 
- Obtained by converting the time-based signal into the freq. domain using Fourier Transform [1]
- Fundamental freq., freq. components, spectral centroid, spectral flux, spectral density, spectral roll-off (used to identify notes, pitch, rhythm and melody) [1]

## Temporal Features: 
- Energy of a signal, zero crossing rate, max. amp, min. energy [1]

## Short Term Features: 
1. Mel Frequency Cepstral Coefficients (MFCC)
    a. Small set of features that describe the shape of the spectral envelope [2]
    b. Used to describe timbre (tone quality; perceived sound quality of a note through different materials)
    c. Involves two changes; from time to frequency domain and back
    d. MFCCs can catch unnatural transitions or frequency components
    e. Can catch differences between natural human pitches, transitions, or inconsistencies in the audio
	
    Method:
    1. Log power spectrum is generated from the audio input before being resampled on the mel scale - this creates discrete cosine transform 
    2. We can compare the DCT of log(abs(FT)) [FT is frequency domain] to the autocorrelation (i.e. clapping in an empty room and comparing the echos to the original clap; similar to amplifying the echoes to find the original signal)
    3. Essentially the DCT of log(abs(ft)) acts as autocorrelation
    4. Why DCT over autocorrelation? - robust to the removal of the fundamental frequency
    5. MEL scale resampling - dimensional reduction and results in 16 values
    6. Interpretation: MFCCs of an audio file can be interpreted as the highpass filtered file’s autocorrelation with the musical pitch removed and robust to bandwidth filtration [2]

2. Spectral Centroid [3]
    a. Acoustic descriptor of timbre; obtained through the weighted average on the frequency rep. Of the signal
    b. Corresponds with a specific type of estimation of the spectral shape/curve
    c. Estimates the center of mass of the spectrum; value corresponds to a frequency on both sides of which the energy is equally distributed
    d. Deep fakes may show unnatural spectral centroids or have strange spikes in varying samples

3. Zero-crossing Rate [4]
    a. Provides info about the number of zero-crossings present in a given signal
        i. ZCR provides indirect information about the frequency content of the signal (i.e. more crossings = higher frequency information)
    b. Can be used to distinguish between voiced and unvoiced regions
    c. Sharp changes in the ZCR may indicate deep faked audio

## Long Term Features:
1. Long-Term Average Spectrum [5]
    a. A fast Fourier transform-generated power spectrum of the frequencies comprising a speech sample
    b. Holds a promise as an acoustic index of voice quality
    c. Relatively weak harmonic energy in higher frequencies of speech spectrum and increase in spectral tilt correspond to breathy or hypo-functional signals
    d. Excessive vocal fold impact and turbulent noise are associated with relatively greater energy in the higher frequencies of the speech spectrum
    e. Sharp/sudden changes in frequencies may indicate deep fakes/synthesized audio 

2. Harmonicity [6]
    a. Degree of acoustic periodicity
    b. Also known as the Harmonics-to-Noise Ratio and is expressed in Db
    c. If 99% of the energy of the signal is in the periodic part, then the HNR is 10*log10(99/1)
    d. An HNR of 0db means there is equal energy in the harmonics and noise
    e. HNR can measure the noisiness of an audio sample; could be used for pattern recognition of deep fakes
    f. Low harmonicity might indicate a lack of natural harmonic structure; common in synthetic audio

3. Formants [7]
    a. Frequency peaks in the spectrum with a high degree of energy
    b. Especially prominent in vowels
    c. Each formant corresponds to a resonance in the vocal tract
    d. Formants can be considered as filters; the filtering of sound takes place in the source sound and causing some frequencies to strengthen while others attenuate
    e. Deep fakes may generate unnatural formant shifts; large spikes or drops

## Libraries
1. Librosa
    a. Can convert audio sample data to numerical values
    b. Audio analysis (has visualization functions)
    c. Can manipulate and extract audio files
    d. Integrates with numpy and matplotlib
    e. Can calculate spectral centroid, spectral roll-off, and zero-crossing rate
https://www.analyticsvidhya.com/blog/2024/01/hands-on-guide-to-librosa-for-handling-audio-files/#:~:text=Librosa%20simplifies%20working%20with%20audio,classification%20and%20audio%20source%20separation.
https://medium.com/@varunreddy1268/audio-data-to-numerical-data-cleaning-short-term-fourier-transforms-mel-spectrograms-7108bb2d780b#:~:text=librosa.,data%20file%20to%20numerical%20data.&text=Sample%20rate%3A%20The%20number%20of,to%20as%20the%20sampling%20rate. 

2. PyAudioAnalysis
    a. Open-source library that provides a range of audio analysis procedures
    b. Contains feature extraction, classification of audio signals, etc.
    c. Used previously for audio event detection, speech emotion recognition, music segmentation, etc.
    d. Can also be used for MFCC
https://pmc.ncbi.nlm.nih.gov/articles/PMC4676707/
Library: https://github.com/tyiannak/pyAudioAnalysis 

3. Python Speech Features
    a. Computes MFCCs, log power spectrum, and other speech features (could be used for formants, harmonicity and long-term average spectrum
https://python-speech-features.readthedocs.io/en/latest/ 

[1] https://www.researchgate.net/post/What-are-the-Spectral-and-Temporal-Features-in-Speech-signal
[2] https://medium.com/@derutycsl/intuitive-understanding-of-mfccs-836d36a1f779
[3] https://timbreandorchestration.org/writings/timbre-lingo/2019/3/29/spectral-centroid
[4]https://vlab.amrita.edu/?sub=3&brch=164&sim=857&cnt=1#:~:text=Short%20Term%20Zero%20Crossing%20Rate%20(ZCR),the%20frame%20size%20as%20shift.
[5]https://pmc.ncbi.nlm.nih.gov/articles/PMC5800529/#:~:text=The%20Long-Term%20Average%20Spectrum,%2C%20&%20Buder%2C%202005).
[6] https://www.fon.hum.uva.nl/praat/manual/Harmonicity.html
[7]https://www.sciencedirect.com/topics/medicine-and-dentistry/formant#:~:text=Formants%20are%20frequency%20peaks%20in,a%20formant%20every%201000%20Hz.

