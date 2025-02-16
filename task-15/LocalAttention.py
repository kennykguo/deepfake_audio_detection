import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram

class LocalAttention(nn.Module):
    def __init__(self, embed_size, num_heads, window_size):
        super(LocalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads)
        self.window_size = window_size

    def forward(self, x):
        batch_size, seq_len, embed_size = x.size()
        output = torch.zeros_like(x)
        for i in range(0, seq_len, self.window_size):
            end = min(i + self.window_size, seq_len)
            attn_output, _ = self.attention(x[:, i:end, :], x[:, i:end, :], x[:, i:end, :])
            output[:, i:end, :] = attn_output
        return output