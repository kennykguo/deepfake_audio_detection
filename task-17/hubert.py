# Step 1: Install Required Libraries
# Run this command in your terminal or notebook before executing the script:
# pip install transformers datasets soundfile

import soundfile as sf
import torch
from transformers import AutoProcessor, HubertModel

# Step 2: Load the HuBERT Model and Processor
def load_hubert_model():
    """
    Load the HuBERT model and processor.
    """
    model_name = "facebook/hubert-large-ls960-ft"  # Pre-trained HuBERT model
    processor = AutoProcessor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    return processor, model

# Step 3: Prepare Audio Data
def load_audio(file_path):
    """
    Load an audio file and extract the waveform and sample rate.
    """
    waveform, sample_rate = sf.read(file_path)
    return waveform, sample_rate

# Step 4: Process Audio and Extract Embeddings
def extract_embeddings(processor, model, waveform, sample_rate):
    """
    Process the audio waveform and extract embeddings using HuBERT.
    """
    # Preprocess the audio
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

    # Pass the processed audio through the HuBERT model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the last hidden states (embeddings)
    embeddings = outputs.last_hidden_state
    return embeddings

# Step 5: Main Function
def main():
    # Load the HuBERT model and processor
    processor, model = load_hubert_model()

    # Load an audio file (replace with your audio file path)
    audio_file_path = "path/to/your/audio.wav"
    waveform, sample_rate = load_audio(audio_file_path)

    # Extract embeddings from the audio
    embeddings = extract_embeddings(processor, model, waveform, sample_rate)

    # Print the shape of the embeddings
    print(f"Embeddings shape: {embeddings.shape}")  # (batch_size, sequence_length, hidden_size)

    # Example: Access the embeddings for further processing
    # embeddings is a tensor of shape (1, sequence_length, 1024) for hubert-large-ls960-ft
    # You can use these embeddings for tasks like classification, clustering, etc.

# Run the script
if __name__ == "__main__":
    main()