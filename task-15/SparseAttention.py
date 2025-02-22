import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram

class SparseAttention(nn.Module):
    def __init__(self, embed_size, num_heads, sparsity_factor):
        super(SparseAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads)
        self.sparsity_factor = sparsity_factor

    def forward(self, x):
        batch_size, seq_len, embed_size = x.size()
        mask = torch.rand(seq_len, seq_len) < self.sparsity_factor
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        return attn_output