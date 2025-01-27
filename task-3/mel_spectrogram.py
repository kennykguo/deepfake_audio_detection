import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import argparse


# Plots waveform, stft, and mel (assuming stereo — only the first channel is plotted)
def plot_waveform(waveform):
    """Plots the waveform of the audio."""
    plt.figure(figsize=(10, 3))
    plt.title('Waveform')
    plt.plot(waveform[0].t().numpy())
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def plot_stft(stft_result, sample_rate, hop_length):
    num_frames = stft_result.shape[-1]
    magnitude = stft_result.abs()
    time = torch.arange(0, num_frames) * hop_length / sample_rate  # Convert frames to time
    plt.figure(figsize=(10, 4))
    plt.imshow(magnitude[0].log2().detach().numpy(), aspect='auto', origin='lower', cmap='inferno', extent=[time[0], time[-1], 0, magnitude[0].shape[0]])
    plt.title('STFT Magnitude')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [bins]')
    plt.show()

def plot_mel_spectrogram(mel_result, sample_rate, hop_length):
    num_frames = mel_result.shape[-1]
    time = torch.arange(0, num_frames) * hop_length / sample_rate  # Convert frames to time
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_result[0].log2().detach().numpy(), aspect='auto', origin='lower', cmap='inferno', extent=[time[0], time[-1], 0, mel_result[0].shape[0]])
    plt.title('Mel Spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Mel bins')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process and visualize audio features')
    parser.add_argument('audio_file', type=str, help='Path to the audio file')
    args = parser.parse_args()

    # Load the audio file
    waveform, sample_rate = torchaudio.load(args.audio_file)

    HOP_LENGTH = 160
    N_FFT = 400
    N_MELS = 23
    # STFT and Mel Spectrogram transforms
    stft_transform = T.Spectrogram(n_fft=N_FFT, win_length=None, hop_length=HOP_LENGTH, power=None)
    stft_result = stft_transform(waveform)

    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_result = mel_transform(waveform)

    # Plot each visualization
    plot_waveform(waveform)
    plot_stft(stft_result, sample_rate, HOP_LENGTH)
    plot_mel_spectrogram(mel_result, sample_rate, HOP_LENGTH)

if __name__ == "__main__":
    main()