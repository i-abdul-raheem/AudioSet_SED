import os
import numpy as np
import matplotlib.pyplot as plt

# Paths
NPY_FILE_PATH = './data/spectrograms_labels.npy'
SPECTROGRAM_SAVE_DIR = './data/spectrograms/'

# Ensure the save directory exists
os.makedirs(SPECTROGRAM_SAVE_DIR, exist_ok=True)

# Load the .npy file
X_train, y_train = np.load(NPY_FILE_PATH, allow_pickle=True)

# Function to save spectrogram as an image
def save_spectrogram(spectrogram, file_path):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

# Iterate over the spectrograms and save them as images
for i, spectrogram in enumerate(X_train):
    # Remove the channel dimension if it exists
    if spectrogram.ndim == 3:
        spectrogram = np.squeeze(spectrogram, axis=-1)
    
    # Save the spectrogram as a PNG image
    save_path = os.path.join(SPECTROGRAM_SAVE_DIR, f'spectrogram_{i}.png')
    save_spectrogram(spectrogram, save_path)

print(f"Spectrogram images saved to {SPECTROGRAM_SAVE_DIR}")
