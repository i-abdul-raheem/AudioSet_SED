import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm
import argparse

# Paths to the dataset
CLASS_LABELS_PATH = './data/class_labels_indices.csv'
TRAIN_METADATA_PATH = './data/train.csv'
AUDIO_DIR = './data/audio/'
X_TRAIN_FILE_PATH = './data/X_train.npy'
Y_TRAIN_FILE_PATH = './data/y_train.npy'
MODEL_SAVE_PATH = 'sed_model.keras'  # Changed to .keras

# Ensure the save directory exists
os.makedirs(os.path.dirname(X_TRAIN_FILE_PATH), exist_ok=True)

# 1. Data Preprocessing
print("Loading class labels and metadata...")

# Load class labels
class_labels = pd.read_csv(CLASS_LABELS_PATH)

# Load training metadata
train_metadata = pd.read_csv(TRAIN_METADATA_PATH)

# Function to extract labels from metadata
def extract_labels(label_str):
    return label_str.split(',')

# Add a column for extracted labels in train metadata
train_metadata['labels'] = train_metadata['positive_labels'].apply(extract_labels)

# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit(train_metadata['labels'])

# Function to load an audio file and extract Mel spectrogram features
def extract_features(file_path, sr=22050, n_mels=128, hop_length=512):
    y, sr = librosa.load(file_path, sr=sr)
    if len(y) == 0:
        return None  # Skip empty audio files
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

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Process and train a sound event detection model.")
parser.add_argument('--cached', action='store_true', help="Use cached data if available.")
args = parser.parse_args()

# Check if cached data exists and if the --cached argument was provided
if args.cached and os.path.exists(X_TRAIN_FILE_PATH) and os.path.exists(Y_TRAIN_FILE_PATH):
    print("Loading cached data...")
    X_train = np.load(X_TRAIN_FILE_PATH)
    y_train = np.load(Y_TRAIN_FILE_PATH)
else:
    # Define a fixed max length for the time dimension (e.g., 400)
    max_length = 400
    input_shape = (128, max_length, 1)  # (n_mels, time, 1)
    
    # 3. Data Processing and Saving Spectrograms
    print("Processing and saving spectrograms...")

    # Prepare training data with progress bar
    X_train = []
    y_train = []

    for i, row in tqdm(train_metadata.iterrows(), total=len(train_metadata), desc="Processing Audio Files"):
        file_path = os.path.join(AUDIO_DIR, row['YTID'] + '.wav')
        features = extract_features(file_path)
        
        if features is None:
            continue  # Skip if the audio file was empty
        
        # Pad or truncate the spectrogram to a fixed length
        features = pad_or_truncate(features, max_length)
        
        # Ensure the spectrogram has the right dimensions
        if features.ndim == 2:
            features = np.expand_dims(features, axis=-1)
        
        X_train.append(features)
        y_train.append(row['labels'])

    X_train = np.array(X_train)
    y_train = mlb.transform(y_train)  # Convert labels to binary format

    # Save X_train and y_train as separate .npy files
    np.save(X_TRAIN_FILE_PATH, X_train)
    np.save(Y_TRAIN_FILE_PATH, y_train)

    print(f"Spectrograms saved to {X_TRAIN_FILE_PATH}")
    print(f"Labels saved to {Y_TRAIN_FILE_PATH}")

# 2. Model Building
print("Building the model...")

def build_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define a fixed max length for the time dimension (e.g., 400)
max_length = 400
input_shape = (128, max_length, 1)  # (n_mels, time, 1)
num_classes = len(mlb.classes_)

# Build the model
model = build_model(input_shape, num_classes)
model.summary()

# Training the model
print("Training the model...")

# Callbacks
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1, callbacks=[checkpoint, reduce_lr])

print(f"Model saved to {MODEL_SAVE_PATH}")

print("Made with ❤️🖤❤️ by Abdul Raheem (https://github.com/i-abdul-raheem)")
