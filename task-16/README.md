# **SincNet vs. LEAF: Feature Extraction Pipeline**

This README provides a pipeline to compare **SincNet** and **LEAF** for feature extraction from raw audio. The goal is to determine which method provides better accuracy for classification tasks.

---

## **1. Pipeline Overview**

| **Pipeline Step**         | **Technology** |
|---------------------------|---------------|
| **Data Loading**          | Torchaudio, Librosa |
| **Feature Extraction**    | SincNet (Custom PyTorch Module), LEAF (DeepMind) |
| **Model Training**        | PyTorch (MLP, CNN) |
| **Evaluation**            | PyTorch Metrics, Matplotlib |
| **Logging & Debugging**   | TensorBoard, Weights & Biases |

---

## **2. Feature Extraction Methods**

### **SincNet**
- Uses **sinc-based band-pass filters** with learnable cutoff frequencies.
- Optimized for **speaker recognition**.
- **Efficient** due to fewer parameters.
- Less adaptive but **robust to noise**.

### **LEAF**
- Uses **fully learnable Gabor filters**.
- More flexible and adapts to **various audio tasks**.
- Computationally heavier but provides **better generalization**.

---

## **3. Accuracy Comparison**

| **Metric**          | **SincNet ✅** | **LEAF ✅** |
|--------------------|:-------------:|:----------:|
| **Speaker Recognition Accuracy** | ✅ Higher | ❌ Slightly lower |
| **Speech Recognition Accuracy**  | ❌ Lower | ✅ Higher |
| **General Audio Classification** | ❌ Limited | ✅ Works across tasks |
| **Noise Robustness** | ✅ More resistant | ❌ More sensitive |

### **Detailed Accuracy Comparisons**

1. **Speaker Recognition Accuracy**
   - **SincNet:** Performs better for speaker identification and verification due to its structured filter design that captures **speaker-specific spectral characteristics**.
   - **LEAF:** Can still perform speaker recognition but may be **less optimized** for this specific task compared to SincNet.

2. **Speech Recognition Accuracy**
   - **SincNet:** Struggles with complex speech recognition tasks where words, phonemes, and intonation patterns vary significantly.
   - **LEAF:** Learns adaptable filter banks, making it superior for **speech-to-text, keyword spotting, and general spoken word classification**.

3. **General Audio Classification**
   - **SincNet:** Less effective for non-speech tasks like **music genre classification, environmental sound recognition**, etc.
   - **LEAF:** More **generalizable across different types of audio** beyond speech, as it learns frequency representations dynamically.

4. **Noise Robustness**
   - **SincNet:** More robust to noise because its **predefined band-pass filters** remain stable even in noisy environments.
   - **LEAF:** More sensitive to noise as its **learned filters may overfit to irrelevant frequency components**, requiring additional regularization.

---

## **4. Conclusion**
- **SincNet** is better for tasks requiring **speaker recognition and noise robustness**.
- **LEAF** is superior for **speech recognition, music, and general audio classification**.
