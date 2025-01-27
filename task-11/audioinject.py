"""
This module implements noise injection to add background noise or distortions to audio samples, simulating real-world conditions for robust model training.
"""

import librosa
import librosa.display
import numpy as np
import soundfile as sf

# load audio file
audio_file = '.wavfile' # replace with .wav file destination
y, sr = librosa.load(audio_file, sr=None)

# add white noise
noise = np.random.normal(0, 0.02, len(y))  
y_noisy = y + noise

# save noisy audio
sf.write('test_audio_noisy.wav', y_noisy, sr)

# plot original and noisy audio
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.5, label='Original')
librosa.display.waveshow(y_noisy, sr=sr, alpha=0.5, color='r', label='Noisy')
plt.legend()
plt.title('Original vs Noisy Audio')
plt.show()