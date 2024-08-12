import os
import numpy as np
import librosa
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer

# Paths
CLASS_LABELS_PATH = './data/class_labels_indices.csv'
MODEL_SAVE_PATH = 'sed_model.keras'
AUDIO_DIR = './data/test/'
PREDICTIONS_DIR = './data/predictions/'

# Ensure the predictions directory exists
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Load class labels and model
print("Loading class labels and model...")
class_labels = pd.read_csv(CLASS_LABELS_PATH)
mlb = MultiLabelBinarizer()
mlb.fit([label for sublist in class_labels['positive_labels'].apply(lambda x: x.split(',')) for label in sublist])

model = load_model(MODEL_SAVE_PATH)

# Function to extract features from an audio file
def extract_features(file_path, sr=22050, n_mels=128, hop_length=512):
    y, sr = librosa.load(file_path, sr=sr)
    if len(y) == 0:
        return None
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

# Function to pad or truncate spectrograms to a fixed length
def pad_or_truncate(spectrogram, max_length):
    if spectrogram.shape[1] > max_length:
        return spectrogram[:, :max_length]
    elif spectrogram.shape[1] < max_length:
        pad_width = max_length - spectrogram.shape[1]
        return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return spectrogram

# Define a fixed max length for the time dimension (e.g., 400)
max_length = 400
input_shape = (128, max_length, 1)  # (n_mels, time, 1)

# Process all audio files in the specified directory
for file_name in os.listdir(AUDIO_DIR):
    if file_name.endswith('.wav'):
        audio_file = os.path.join(AUDIO_DIR, file_name)
        print(f"Processing file: {audio_file}")

        # Extract features
        features = extract_features(audio_file)
        if features is None:
            print(f"Skipping empty or unreadable file: {audio_file}")
            continue
        
        # Pad or truncate the spectrogram
        features = pad_or_truncate(features, max_length)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = np.expand_dims(features, axis=-1)  # Add channel dimension
        
        # Predict
        predictions = model.predict(features)
        predicted_labels = mlb.inverse_transform(predictions > 0.5)

        # Save predictions
        base_name = os.path.basename(file_name)
        prediction_file = os.path.join(PREDICTIONS_DIR, f"{os.path.splitext(base_name)[0]}_predictions.txt")
        
        with open(prediction_file, 'w') as f:
            for label in predicted_labels[0]:
                f.write(f"{label}\n")
        
        print(f"Predictions saved to {prediction_file}")

print("Prediction completed!")
