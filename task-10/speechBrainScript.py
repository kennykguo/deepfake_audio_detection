import torchaudio
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.features import MFCC

# Load the audio file
audio_file = '.wav' # path to the .wav audio file
signal = read_audio(audio_file)

# Initialize the MFCC feature extractor
# num_mfcc: Number of MFCC coefficients to extract (e.g. 30)
num_mfcc = 30
mfcc_extractor = MFCC(num_mfcc)

# Extract MFCC features
mfcc_features = mfcc_extractor(signal)

# Print the extracted MFCCs
print(mfcc_features.shape)  # (time_steps, num_mfcc)

# The 'mfcc_features' tensor contains the extracted MFCCs,
# which can be further used in deep fake audio detection models.