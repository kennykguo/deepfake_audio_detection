Explain the Mel, Gammatone, and discrete cosine transform auditory filters, detailing their roles in audio feature extraction. Give example input and output tensor shapes for each filter. Write a sample script showcasing uses of these auditory filters.

# Auditory Filters

## Overview
Auditory filters are devices that **boost certain frequencies** and **attenuate others** to mimic human hearing. These filters help process and analyze sound signals efficiently.

---

## Mel Filter Bank
### Overview
- **Mel** stands for **Melody** (focused on pitch perception).
- Commonly used as **Mel-Frequency Cepstral Coefficients (MFCCs)**.
- The **Mel-scale** is **logarithmic**, assuming human perception of sound is non-linear.
- **Higher frequencies** are spaced farther apart because humans are more sensitive to lower frequencies.
- **Triangular filters** are used to capture different **Mel-scaled** frequencies.

### How to Use Mel
1. Convert frequencies to the **Mel scale**.
2. Choose the number of **Mel bands** (typically **40~128**).
3. Construct **Mel filter banks**:
   - Convert lowest & highest frequency to Mel => **L & H**.
   - Create equally spaced points between **L & H**.
   - Determine **center frequencies** of different Mel bands.
   - Convert points **back to Hertz**.
   - Round to the **nearest frequency bin**.
4. Create **triangular filters**:
   - Center point as **vertex**.
   - Lower end = **center of previous Mel band**.
   - Higher end = **center of next Mel band**.
   - Ends where the weight is **zero** (no contribution from that frequency bin).
   - **M = (# bands, framesize/2 + 1)**.

5. **Apply Mel filter banks to the spectrogram**:
   - **Y = (framesize / 2 + 1, # frames)**.
   - **Mel spectrogram** is computed as: `Mel Spectrogram = M * Y` (matrix multiplication).
   - Final **shape**: `(# bands, # frames)`.

**Further Reading:**  
- [Mel Spectrograms Explained](https://www.mathworks.com/help/audio/ref/melspectrogramblock.html)

---

## Gammatone Filter Bank
### Overview
- Commonly used as **Gammatone Frequency Cepstral Coefficients (GFCCs)**.
- Simulates **cochlear filtering**, making it more biologically plausible than the **Mel scale**.
- Uses **bandpass linear filters** with an **impulse response**.
- Processes **time-varying input signals** to produce output signals.
- Based on the **Gamma function**.
- Time-domain impulse response: **sinusoid (Linear Time-Invariant, LTI)**.
- **Goal**: Realize a particular **frequency response**.
- Uses **Equivalent Rectangular Bandwidth (ERB) scale**.

**Further Reading:**  
- [Gammatone Filterbank - MathWorks](https://www.mathworks.com/help/audio/ref/gammatonefilterbank-system-object.html)  
- [Gammatonegram MATLAB Resource](https://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/5)

---

## Discrete Cosine Transform (DCT)
### Overview
- **DCT** is used to selectively manipulate different frequencies within an audio signal.
- Stores important **coefficients** based on **human perception** & desired **audio quality**.
- **DCT is applied after spectral transformations** (during **MFCC processing**), not directly on raw audio.
- **Transforms time-domain** signals into **frequency-domain** components.
- **Inverse DCT (IDCT)** is used to reconstruct audio signals with minimal loss.
- **Compression of frequency features** helps in reducing redundancy.
- **Modified DCT (MDCT)** is widely used in audio processing.

**Further Reading:**  
- [DCT - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/016516849090115F) *(German)*  
- [DCT - Wikipedia](https://en.wikipedia.org/wiki/Discrete_cosine_transform)  
- **The Discrete Cosine Transform** [7:38~]  
- **Audio Coding - MDCT Implementation**

---

## Input & Output Tensor Shapes
| Transformation | Input Shape | Output Shape |
|---------------|------------|-------------|
| **Mel** | (32, 100, 1024) | (32, 100, 40) |
| **Gammatone** | (32, 100, 1024) | (32, 100, 64) |
| **DCT** | (32, 100, 40) | (32, 100, 13) |

---
