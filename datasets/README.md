
# Dataset downloadable from Kaggle
[Audio Dataset](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition/data)
- Divided into 2 folders: Real & Fake.
- Real folder contains 8 Original speeches by 8 well-known influential people.
- Fake folder stores audio that were converted into another person's voice, as denoted in each file's name.
  - Background noise was removed.
  - 56 files
- .wav audio files, 10 minutes each.

# ASVspoof
[Database](https://www.asvspoof.org/database)
- Automatic Speaker Verification is used for biometric identification.
- Spoofing refers to presentation attacks
  - Impersonation of something or someone to gain access to the system

## ASVspoof 2015
- Focus on detecting text-to-speech synthesis (TTS) and voice conversion (VC) attacks.

## ASVspoof 2017
- Focus on replay attacks
  - Attacker intercepts & retransmits a perviously sent data packet to gain unauthorized access.

## ASVspoof 2019
[Paper](https://arxiv.org/pdf/1911.01601) ||
[Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)
- Spoofing attacks within a logical access (LA) scenario using the latest TTS and VS technologies, including state-of-the art neural acoustic & waveform model techniques.
  - TTS, VC, replay attacks considered
- Carefully controlled simulations of replayed speech
- LA Control: remote user seeks access to a system/service protected by ASV
- PA Control: use of ASV to protect access to a sensitive/secure physical space or facility
### Database Partitions
- Based on Voice Cloning Tookit (VCTK): multi-speaker English speech database recorded in a hemi-anechoic chamber
  - Sampling rate of 96 kHz downsampled to 16 kHz
  - 107 speakers (46 M, 61 F)
  - Training => 20 speakers (8 M, 12 F)
  - Development => 10 target & 10 non-target speakers (4 M, 6 F)
  - Evaluation => 48 target & 19 non-target speakers (21 M, 27 F)
- .flac files in train, dev, and eval folders.

## ASVspoof 2021
[Paper](https://www.isca-archive.org/asvspoof_2021/yamagishi21_asvspoof.pdf) || [Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/avsspoof-2021)
- DF Control: attacker has access to voice recordings of a targeted victim
- Addition of DeepFake evaluation data (source from ASVspoof 2019)
