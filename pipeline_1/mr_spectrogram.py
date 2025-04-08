import torch
import numpy
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

# waveforms - processed (B, 80000) data torch tensor
class MRSpecDataset(Dataset):
    def __init__(self,
                 waveforms,
                 sr=16000):
        self.sr = sr  # Sampling rate
        self.config = {
            'short': {'n_fft': 256, 'win_length': 160, 'hop_length': 80},
            'medium': {'n_fft': 512, 'win_length': 400, 'hop_length': 160},
            'long': {'n_fft': 1024, 'win_length': 800, 'hop_length': 320}
        }
        
        short = self.compute_stft(waveforms, self.config['short'])
        medium = self.compute_stft(waveforms, self.config['medium'])
        long = self.compute_stft(waveforms, self.config['long'])
        # TODO: Convert these further into mel & phase spectrograms and stack
    
    def compute_stft(self, waveforms, config): 
        # Could add padding/normalization using additional arguments listed.
        transform = T.Spectrogram(n_fft=config['n_fft'], win_length=config['win_length'], hop_length=config['hop_length'],
                                  power=None, center=True, normalized=False, pad=0)      
        return transform(waveforms) 
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

# Example Usage
waveform = torch.randn(10, 80000) # (Batch, Samples)
dataset = MRSpecDataset(waveform)