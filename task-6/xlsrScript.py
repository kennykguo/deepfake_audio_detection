import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torchaudio
import librosa
import numpy as np

"""
Note: The following python script is written with the assumption that the XLS-R model has already been fine-tuned 
for deepfake audio detection through methods similar to the SLS Classifier as discussed in the README.md. The script extracts the highest confidence value of the output 
of the fine-tuned XLS-R model. 
"""


# Utilizing the most basic, facebook xls-r model
model_name = "facebook/xls-r-1b"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()


def preprocess_audio(audio_path):
    # Load audio file using librosa or torchaudio
    # Sample audio at a 16khz rate
    audio, sr = librosa.load(audio_path, sr=16000)

    return torch.tensor(audio).unsqueeze(0)


def detect_deepfake(audio_path):
    # Preprocess the audio
    audio_input = preprocess_audio(audio_path)

    # Pass the audio through the model's processor
    # Returns dictionary in PyTorch Tensor format
    inputs = processor(audio_input, return_tensors="pt", padding=True)

    # Forward pass to get logits (raw predictions)
    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits to probabilities (Softmax)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted class (deepfake or real)
    prediction = torch.argmax(probs, dim=-1).item()

    return "Deepfake" if prediction == 1 else "Real"


# Run the model
audio_file = "(insert your audio here).wav"
result = detect_deepfake(audio_file)
print(f"Deepfake Audio Detection Result: {result}")
