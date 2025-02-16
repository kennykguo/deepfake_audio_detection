import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader, random_split

# Define the dataset
def prepare_datasets():
    # Download and load the dataset
    train_dataset = LIBRISPEECH(root="data", url="train-clean-100", download=True)
    test_dataset = LIBRISPEECH(root="data", url="test-clean", download=True)

    # Split the training dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset

# Prepare the datasets
train_dataset, val_dataset, test_dataset = prepare_datasets()

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Assuming LocalAttention and SparseAttention are defined in separate files
from LocalAttention import LocalAttention
from SparseAttention import SparseAttention

class AudioDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.mel_spectrogram = MelSpectrogram()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, _, _, label, _ = self.dataset[idx]
        mel_spectrogram = self.mel_spectrogram(waveform)
        mel_spectrogram = mel_spectrogram.permute(0, 2, 1)  # (batch_size, seq_len, feature_dim)
        return mel_spectrogram, label

class TransformerLocalAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, window_size):
        super(TransformerLocalAttention, self).__init__()
        self.embedding = nn.Linear(128, embed_size)
        self.local_attention_layers = nn.ModuleList(
            [LocalAttention(embed_size, num_heads, window_size) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_size, 10)  # Example output size

    def forward(self, x):
        x = self.embedding(x)
        for local_layer in self.local_attention_layers:
            x = local_layer(x)
        x = self.fc(x.mean(dim=1))
        return x

class TransformerSparseAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, sparsity_factor):
        super(TransformerSparseAttention, self).__init__()
        self.embedding = nn.Linear(128, embed_size)
        self.sparse_attention_layers = nn.ModuleList(
            [SparseAttention(embed_size, num_heads, sparsity_factor) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_size, 10)  # Example output size

    def forward(self, x):
        x = self.embedding(x)
        for sparse_layer in self.sparse_attention_layers:
            x = sparse_layer(x)
        x = self.fc(x.mean(dim=1))
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    # Prepare the datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets()

    # Create data loaders
    train_dataloader = DataLoader(AudioDataset(train_dataset), batch_size=32, shuffle=True)
    val_dataloader = DataLoader(AudioDataset(val_dataset), batch_size=32, shuffle=False)
    test_dataloader = DataLoader(AudioDataset(test_dataset), batch_size=32, shuffle=False)

    # Initialize the models, criterion, and optimizer
    model_local = TransformerLocalAttention(embed_size=256, num_heads=8, num_layers=4, window_size=10)
    model_sparse = TransformerSparseAttention(embed_size=256, num_heads=8, num_layers=4, sparsity_factor=0.1)
    criterion = nn.CrossEntropyLoss()
    optimizer_local = optim.Adam(model_local.parameters(), lr=0.001)
    optimizer_sparse = optim.Adam(model_sparse.parameters(), lr=0.001)

    # Train the local attention model
    print("Training Local Attention Model")
    train_model(model_local, train_dataloader, criterion, optimizer_local, num_epochs=10)
    accuracy_local, precision_local, recall_local, f1_local = evaluate_model(model_local, test_dataloader)
    print(f'Local Attention Model - Accuracy: {accuracy_local:.4f}, Precision: {precision_local:.4f}, Recall: {recall_local:.4f}, F1 Score: {f1_local:.4f}')

    # Train the sparse attention model
    print("Training Sparse Attention Model")
    train_model(model_sparse, train_dataloader, criterion, optimizer_sparse, num_epochs=10)
    accuracy_sparse, precision_sparse, recall_sparse, f1_sparse = evaluate_model(model_sparse, test_dataloader)
    print(f'Sparse Attention Model - Accuracy: {accuracy_sparse:.4f}, Precision: {precision_sparse:.4f}, Recall: {recall_sparse:.4f}, F1 Score: {f1_sparse:.4f}')