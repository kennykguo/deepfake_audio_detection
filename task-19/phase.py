import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process and visualize audio features')
    parser.add_argument('audio_file', type=str, help='Path to the audio file')
    args = parser.parse_args()

    # Load the audio file
    waveform, sample_rate = torchaudio.load(args.audio_file)

    HOP_LENGTH = 160
    N_FFT = 400
    
    # STFT 
    stft_transform = T.Spectrogram(n_fft=N_FFT, win_length=None, hop_length=HOP_LENGTH, power=None)
    stft_result = stft_transform(waveform)
    
    # Magnitude is the strength of frequencies over time
    # Phase reflects timing/arrangement of frequencies (when & how a frequency appears)
    # Phase information is lost in Mel Spectrograms
    phase = stft_result.angle() # (in radians)
    magnitude = stft_result.abs()

    # concat magnitude and phase information
    magphase = torch.cat([magnitude, phase], dim=1)
    print(magphase.shape)

if __name__ == "__main__":
    main()