# What is Whisper?

- **Automatic Speech Recognition (ASR) system**
- **Sequence-to-sequence**
- **Speech Translation model**
- **Trained on multilingual & multitask supervised web data**

## Architecture
- **End-to-end approach**
- **Encoder-decoder Transformer**

## Features
- **Robust Speech Recognition in both structured & unstructured data**
  - Accents
  - Background noise
  - Technical language
- **Language Identification**
- **Speech-to-Text Translation**
- **Multilingual Transcription**
- **Multilingual Translation**
- **Text Prompts for better focusing on user’s needs**
- **Various versions**

## Use Cases
- **Captioning/Transcription, real-time & recorded audios**
  - Meetings, podcasts, videos, court trials
- **Multilingual Translation**
  - International communication, language learning, global content distribution
- **Customer Support**
  - Chatbots & automated calls with verbal interaction

## Input
- **Audio**
  - Split into <= 30s chunks, sampling rate of 16 kHz
  - Mono-channel audio [VS stereo]
  - Converted into a log-Mel spectrogram
  - Passed into an encoder
- **Preprocessing Audio**
  - Use libraries like FFmpeg (video), Librosa (audio), PyDub (audio: .wav files)

## Output
- **Decoder**
  - Predict corresponding text caption w/ special tokens that tell the model what task to perform
- **Text Transcription**
- **Translation**
- **Timestamps (of when certain words or phrases were spoken)**
- **Spoken Language(s)**
  - Based on segments of audio

## Warning
- **NOT intended for classification**

## Usage for Deepfake Audios
- Seems to be used as feature extractor
- Should be utilized as the Front End of a specific Backend architecture
- Before Post Processing and Encoding
- Used on ASVspoof 2021 (DF) and DeepFakes In-The-Wild datasets

---

# Improved DeepFake Detection Using Whisper Features (Kawa et al., 2023)
- Inferred Whisper would ignore most of naturally occurring artefacts & help identify artificially modified speech samples
- Help with problem of poor efficacy of the models on the data outside of the training set’s distribution => generalization
- Experimented with **tiny.en**
  - Smallest model, only on English, since main language of DF datasets is English
  - 7,632,384 parameters & outputs data of shape (376, 1500)

### Experimental Setup
- **Input**
  - Resampled to 16 kHz mono-channel
  - Removed silences longer than 0.2s
  - Padding filled with speech
    - Typically filled with zeros
  - Trimmed samples to 30s of content
    - Typically 4s, when using other models
- 100,000 training & 25,000 validation samples of ASVspoof 2021
- Single GPU (NVIDIA)
- Learning rate of 10^-4, weight decay 10^-4
- Binary cross-entropy function for 10 epochs, batch size of 8
- **Result using Equal Error Rate (EER) metric**
  - Commonly used in DF and spoofing problems
- Fixed randomness seed => ensure reproducibility
- Testing Whisper’s encoder, better when concatenated with other front-ends like LFCC and MFCC
  - SpecRNet
  - MesoNet
  - LCNN
- Unfrozen Whisper features improved results
- Whisper probably works well with RNN because it extracts prominent attributes that do not tend to be hidden
- Findings suggest a negative impact of Whisper features on other front-ends

---

# Whisper + AASIST for DeepFake Audio Detection (Luo & Sivasundari, 2024)

# Prompt Tuning for Audio Deepfake Detection (Oiso et al., 2024)

# Investigating Prosodic Signatures via Speech Pre-Trained Models for Audio Deepfake Source Attribution (Phukan et al., 2024)