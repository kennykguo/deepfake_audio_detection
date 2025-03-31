# Task
https://docs.google.com/document/d/1K0-FpiUHKOZyu-k23gHuZtLmnLbi9R8hDio8WrHrvy8/edit?tab=t.0
# Read
https://arxiv.org/abs/2104.01778
# Code
https://github.com/YuanGongND/ast
# Notes
- AST applies the **Vision Transformer (ViT) framework** to **log Mel spectrograms** and demonstrates state-of-the-art (SOTA) results.
- **ImageNet pretraining** significantly enhances AST's performance by leveraging transfer learning from vision to audio.
- AST outperforms CNN-based models on *AudioSet, ESC-50, and Speech Commands benchmarks.
- The study confirms the importance of patch embedding, positional encoding adaptation, and pretraining in AST’s success.
## Abstract
- CNNs are widely used for end-to-end audio classification models
- Recent trend of adding a self-attention mechanism to the CNN to form a hybrid model
- The paper focuses on an AST (Audio Spectrogram Transformer) model that doesn't use CNNs
## 1. Introduction
Traditional **deep learning models for audio classification** use **convolutional neural networks (CNNs)**, which process log Mel spectrograms as 2D inputs, extracting hierarchical features with spatial locality. While CNNs are effective, they are **limited in modeling long-range dependencies**, which are critical for recognizing complex audio patterns.

Recent advancements have introduced **hybrid CNN-attention models**, where **self-attention layers** supplement CNN feature extractors to enhance global context learning. While these models improve performance, they still **depend on CNNs** as their primary backbone.
## 2. Audio Spectrogram Transformer
### 2.1 Model Architecture
The **Audio Spectrogram Transformer (AST)** follows a pure Transformer-based approach inspired by the **Vision Transformer (ViT)**. It operates on **log Mel spectrograms**, treating them as 2D inputs.

#### **Key Components of AST:**  
1. **Patch Embedding:**
   - The input spectrogram is divided into 16×16 patche* with an overlap of 6 pixels.
   - Each patch is flattened and linearly projected into a 1D vector representation.

2. **Positional Embeddings:**
   - Since Transformers lack built-in spatial awareness, AST adds learnable positional embeddings to retain spectrogram structure.
   - These embeddings are adapted from ViT via bilinear interpolation.

3. **Multi-Head Self-Attention:**
   - AST consists of **12 layers** of **Transformer encoders**, each featuring **multi-head self-attention (MHSA)** and **feed-forward networks**.
   - MHSA captures **both short-range and long-range dependencies** in the spectrogram, unlike CNNs, which primarily extract local patterns.

4. **Class Token ([CLS] Token):**
   - A **learnable classification token** is prepended to the input sequence.
   - The final **[CLS] embedding** serves as the global representation for classification.

5. **Fully Connected Head:**
   - The output from the **[CLS] token** is passed through a **fully connected (FC) classifier** to generate predictions.

### ImageNet Pretraining
Training Transformers from scratch requires large-scale datasets, which are often unavailable for audio classification. To address this, AST leverages ImageNet-pretrained ViT weights through a cross-domain transfer learning strategy:

- Patch embedding weights from ViT are reused by averaging RGB channels.
- Positional embeddings are resized via bilinear interpolation to fit the spectrogram dimensions.
- Pretrained Transformer weights (excluding the classification layer) are retained.

Pretraining on ImageNet significantly boosts AST’s performance, especially on small datasets like ESC-50. *(Gong et al., 2021)*.
## 3. Experiments
### 3.1 AudioSet Experiments
#### 3.1.1 Dataset and Training Details
- **AudioSet** contains **2 million** 10-second audio clips across **527 classes**.
- The authors evaluate AST on:
  - **Balanced Training Set** (22k samples)
  - **Full Training Set** (2M samples)
  - **Evaluation Set** (20k samples)

Training details:
- **Data augmentation:** Mixup, SpecAugment
- **Optimizer:** Adam
- **Loss function:** Binary cross-entropy
- **Batch size:** 12
- **Learning rate:** 1e-5 (full set), 5e-5 (balanced set)
- **Epochs:** 5 (full set), 25 (balanced set)
#### 3.1.2. AudioSet Results
| Model | Balanced Set mAP | Full Set mAP |
|---|---|---|
| Baseline CNN | - | 0.314 |
| CNN+Attention (PSLA) | 0.319 | 0.444 |
| AST (Single) | **0.347** | **0.459** |
| AST (Ensemble) | **0.378** | **0.485** |

#### 3.1.3. Ablation Study
- ImageNet pretraining increases mAP from **0.366 → 0.459**.
- Patch overlap (6 pixels) boosts performance by reducing information loss.
- Positional embedding adaptation is critical for Transformer generalization.
### 3.2 Results on ESC-50 and Speech Commands
| Model | ESC-50 Accuracy | Speech Commands Accuracy |
|---|---|---|
| CNN-based (SOTA) | 94.7% | 97.7% |
| AST (Single) | 95.6% | 98.1% |
## 4. Conclusions