import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram

# import classes
from LocalAttention import LocalAttention
from SparseAttention import SparseAttention

class Transformer(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, window_size, sparsity_factor):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(128, embed_size)
        self.local_attention_layers = nn.ModuleList(
            [LocalAttention(embed_size, num_heads, window_size) for _ in range(num_layers)]
        )
        self.sparse_attention_layers = nn.ModuleList(
            [SparseAttention(embed_size, num_heads, sparsity_factor) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_size, 10)  # Example output size

    def forward(self, x):
        x = self.embedding(x)
        for local_layer, sparse_layer in zip(self.local_attention_layers, self.sparse_attention_layers):
            x = local_layer(x) + sparse_layer(x)
        x = self.fc(x.mean(dim=1))
        return x

def load_wav(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path, backend='soundfile')
        mel_spectrogram = MelSpectrogram()(waveform)
        return mel_spectrogram
    except RuntimeError as e:
        print(f"Error loading {file_path}: {e}")
        return None

if __name__ == "__main__":
    file_path = "example.wav"
    mel_spectrogram = load_wav(file_path)
    if mel_spectrogram is not None:
        mel_spectrogram = mel_spectrogram.permute(0, 2, 1)  # (batch_size, seq_len, feature_dim)

        model = Transformer(embed_size=256, num_heads=8, num_layers=4, window_size=10, sparsity_factor=0.1)
        output = model(mel_spectrogram)
        print(output)
    else:
        print("Failed to load the audio file.")