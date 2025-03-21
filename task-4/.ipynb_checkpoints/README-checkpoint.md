# Task - Compile a list of three real audio datasets and three fake audio datasets, detailing their contents and usefulness for training and testing the detection model. List the sources for all of them, and if they have papers on them, please list them as well.

# Fake Audio Datasets

## Dataset #1: SceneFake Dataset

### Contents:
#### Fake Audio Dataset
- 10,295 WAV audio files
- Each 2~3 seconds in length
- Only tampered acoustic scenes of real audio
- See task #14 for more details on what SceneFake is

#### Real Audio Dataset
- 2,548 WAV audio files
- Each 2~3 seconds in length

This dataset is special in the sense that it can provide training data for our model to not only focus on the verbal aspect of the audio but also on the audio environment.

**Source:** [Kaggle: SceneFake Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/scenefake)  
**Related Papers:** [SceneFake Paper](https://arxiv.org/pdf/2211.06073)

---

## Dataset #2: DEEP-VOICE; Retrieval-based Voice Conversion

### Contents:
#### Fake Audio Dataset
- 56 WAV audio files
- 8 people: each person converted to the other 7 people

**Celebrity Samples:**
- Joe Biden: 10:00
- Linus Sebastian: 09:30
- Margot Robbie: 01:19
- Elon Musk: 10:00
- Barack Obama: 10:00
- Ryan Gosling: 01:33
- Taylor Swift: 10:00
- Donald Trump: 10:00

- Background noise was removed before conversion, and added back after conversion.
- CSV file containing extracted data from the audio files.

#### Real Audio Dataset
- 8 WAV audio files
- Original audio from each celebrity

The audios in this dataset are full-length, but data is already extracted to speed up the analyzing process.

**Source:** [Kaggle: DEEP-VOICE Dataset](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition/data)  
**Related Papers:** [DEEP-VOICE Paper](https://arxiv.org/pdf/2308.12734)

---

## Dataset #3: Fake-or-Real Dataset - Audio Processing Techniques Lab at York (APTLY)

### Contents:
#### Testing Data
- 2,370 fake WAV audios
- 2,264 real WAV audios

#### Training Data
- 21,717 fake WAV audios
- 26,941 real WAV audios

#### Validation Data
- 5,398 fake WAV audios
- 5,400 real WAV audios

All audios are about a sentence in length, lasting **5~10 seconds**. 

The fake utterances were generated using various **Text-to-Speech (TTS) systems** (see task #14 for more details on TTS).

**Real utterances** are taken from open-source datasets, including:
- [Arctic Dataset](http://festvox.org/cmu_arctic/)
- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [VoxForge Dataset](http://www.voxforge.org)

The version uploaded is the **normalized version**, meaning the audios have been preprocessed for uniformity (e.g., **normalized volume, 16kHz sampling**).

**Source:** [APTLY Dataset](https://bil.eecs.yorku.ca/datasets/)  
**Related Papers:** [Fake-or-Real Paper](https://bil.eecs.yorku.ca/wp-content/uploads/2020/01/FoR-Dataset_RR_VT_final.pdf)
