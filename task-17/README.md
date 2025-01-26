# Task: Research and Document the Conformer, Performer, HuBERT, and Wav2Vec 2.0 Models

## 1. HuBERT (Hidden-unit BERT)
### Overview
HuBERT stands for Hidden Unit BERT, trained similarly to a BERT model by masking random segments of input features and predicting these segments. It is a transformer model designed to extract high-quality speech representations that capture both acoustic and linguistic information. This ability makes it valuable for distinguishing between real and fake audio. HuBERT is pretrained on a large corpus of real speech data and can be fine-tuned on a labeled dataset with both real and deepfake audio samples. Varying clustering during pretraining improves accuracy by enabling the model to learn different granularities in speech.

### Architecture
- **Input**: Raw audio waveform or 2D feature matrix (e.g., log Mel-spectrogram, MFCCs).
  - Shape: `(batch_size, sequence_length)` for raw audio or `(batch_size, bins, frames)` for spectrograms.
- **Feature Extraction**: Converts raw audio to feature representation.
  - Output Shape: `(batch_size, bins, frames)`.
- **Transformer Encoder**: Processes the input features.
  - Output Shape: `(batch_size, frames, hidden_size)`.
- **Transformer Output**: Hidden states are used as embeddings by removing the cluster prediction head.
  - Output Shape: `(batch_size, frames, hidden_size)`.

### Flow:
1. Raw Audio: `(batch_size, sequence_length)`
2. Feature Extraction: `(batch_size, bins, frames)`
3. Transformer Encoder: `(batch_size, frames, hidden_size)`
4. Fine-Tuning/Inference: Embeddings `(batch_size, frames, hidden_size)`

---

## 2. Conformer
### Overview
The Conformer model combines convolutional neural networks (CNNs) and transformers to capture both local and global dependencies in audio signals.

### Architecture
- **Input**: Raw audio waveform or 2D feature matrix (e.g., log Mel-spectrogram).
  - Shape: `(batch_size, sequence_length)` or `(batch_size, bins, frames)`.
- **Feature Extraction**: Converts raw audio to feature representation.
  - Output Shape: `(batch_size, bins, frames)`.
- **Conformer Encoder**: Combines convolutional layers (local patterns) and transformer layers (global context).
  - Output Shape: `(batch_size, frames, hidden_size)`.
- **Task-Specific Head**: Linear layer for classification tasks.
  - Output Shape: `(batch_size, frames, num_classes)`.

### Flow:
1. Raw Audio: `(batch_size, sequence_length)`
2. Feature Extraction: `(batch_size, bins, frames)`
3. Conformer Encoder: `(batch_size, frames, hidden_size)`
4. Task-Specific Head: `(batch_size, frames, num_classes)`

---

## 3. Performer
### Overview
The Performer is a transformer variant that uses efficient attention mechanisms (e.g., FAVOR+ or linear attention) to reduce computational complexity. It is designed to handle long sequences, making it effective for audio tasks.

### Architecture
- **Input**: Raw audio waveform or 2D feature matrix.
  - Shape: `(batch_size, sequence_length)` or `(batch_size, bins, frames)`.
- **Feature Extraction**: Converts raw audio to feature representation.
  - Output Shape: `(batch_size, bins, frames)`.
- **Performer Encoder**: Processes long sequences using efficient attention mechanisms.
  - Output Shape: `(batch_size, frames, hidden_size)`.
- **Task-Specific Head**: Linear layer for classification tasks.
  - Output Shape: `(batch_size, frames, num_classes)`.

### Flow:
1. Raw Audio: `(batch_size, sequence_length)`
2. Feature Extraction: `(batch_size, bins, frames)`
3. Performer Encoder: `(batch_size, frames, hidden_size)`
4. Task-Specific Head: `(batch_size, frames, num_classes)`

---

## 4. Wav2Vec 2.0
### Overview
Wav2Vec 2.0 is a self-supervised learning model for speech representation. It is trained by masking parts of latent speech representations and predicting the quantized versions of these masked regions.

### Architecture
- **Input**: Raw audio waveform.
  - Shape: `(batch_size, sequence_length)`.
- **Feature Encoder**: A CNN extracts latent speech representations.
  - Output Shape: `(batch_size, frames, feature_dim)`.
- **Transformer Encoder**: Processes the latent representations.
  - Output Shape: `(batch_size, frames, hidden_size)`.
- **Quantization Module**: Converts latent representations into discrete units.
  - Output Shape: `(batch_size, frames, num_quantized_units)`.
- **Task-Specific Head**: Linear layer for classification tasks.
  - Output Shape: `(batch_size, frames, num_classes)`.

### Flow:
1. Raw Audio: `(batch_size, sequence_length)`
2. Feature Encoder: `(batch_size, frames, feature_dim)`
3. Transformer Encoder: `(batch_size, frames, hidden_size)`
4. Pretraining: Quantization Module `(batch_size, frames, num_quantized_units)`
5. Fine-Tuning/Inference: Task-Specific Head `(batch_size, frames, num_classes)`
