# %%
import os
import subprocess
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Clone the CREMA-D dataset using Git LFS (use subprocess to capture output)
try:
    clone_result = subprocess.run(["git", "clone", "https://github.com/CheyneyComputerScience/CREMA-D.git"], check=True)
    if clone_result.returncode == 0:
        print("Repository cloned successfully.")
    else:
        print("Repository cloning failed.")

    lfs_install_result = subprocess.run(["git", "lfs", "install"], check=True)
    if lfs_install_result.returncode == 0:
        print("Git LFS installed successfully.")
    else:
        print("Git LFS installation failed.")

    lfs_pull_result = subprocess.run(["git", "lfs", "pull"], cwd="CREMA-D", check=True)
    if lfs_pull_result.returncode == 0:
        print("Git LFS pull completed successfully.")
    else:
        print("Git LFS pull failed.")
except subprocess.CalledProcessError as e:
    print(f"Error while cloning the repository or pulling LFS files: {e}")

# Update the dataset path to check multiple possible folder structures
dataset_paths = [
    "CREMA-D/AudioWAV",
    "CREMA-D/Audio",
    "CREMA-D/audio",
    "CREMA-D/AudioFiles"
]

# Verify if any dataset path exists
dataset_path = None
for path in dataset_paths:
    if os.path.exists(path):
        dataset_path = path
        print(f"Using dataset path: {dataset_path}")
        break

if dataset_path is None:
    raise ValueError(
        f"None of the dataset paths {dataset_paths} exist. Please check the repository structure and try again.")

# Print files in dataset path for verification
print("Files in dataset path:")
found_files = False
for root, _, files in os.walk(dataset_path):
    for file in files:
        print(file)
        found_files = True

if not found_files:
    raise ValueError("No audio files found in the dataset path. Please ensure the dataset path contains .wav files.")


# Prepare the data
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}, {e}")
        return None


# Prepare a dataframe for features
audio_files = []
labels = []

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            label = file.split('_')[2]  # Assuming labels are stored in the filename pattern
            feature = extract_features(file_path)
            if feature is not None:
                audio_files.append(feature)
                labels.append(label)

# Convert to dataframe
if len(audio_files) == 0:
    raise ValueError(
        "No audio features extracted. Please ensure the dataset path is correct and contains valid .wav files.")

data = pd.DataFrame(audio_files)
data['label'] = labels

# Encode labels
encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])

# Split data into training and testing features
X = data.iloc[:, :-1]
y = data['label']

if X.empty:
    raise ValueError("Features data is empty. Please check the feature extraction process.")

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Sequential Neural Network Model
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 50
batch_size = 32

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# %%
