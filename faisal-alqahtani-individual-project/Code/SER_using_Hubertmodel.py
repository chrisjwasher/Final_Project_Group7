import os
import subprocess
import sys
import torch
import librosa
import numpy as np
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import warnings
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Function to check if Git LFS is installed
def check_git_lfs():
    try:
        subprocess.run(["git", "lfs", "version"], check=True, stdout=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

# Function to install Git LFS
def install_git_lfs():
    print("Attempting to install Git LFS...")
    if sys.platform.startswith('linux'):
        subprocess.run(["sudo", "apt-get", "install", "git-lfs"], check=True)
    elif sys.platform == 'darwin':
        subprocess.run(["brew", "install", "git-lfs"], check=True)
    elif sys.platform == 'win32':
        print("Please install Git LFS manually from https://git-lfs.github.com/")
        sys.exit(1)
    else:
        print("Unsupported OS. Please install Git LFS manually.")
        sys.exit(1)
    subprocess.run(["git", "lfs", "install"], check=True)
    print("Git LFS installed successfully.")

# Clone the CREMA-D dataset using Git LFS
try:
    if not check_git_lfs():
        install_git_lfs()
    else:
        print("Git LFS is already installed.")

    # Initialize Git LFS
    subprocess.run(["git", "lfs", "install"], check=True)
    print("Git LFS initialized successfully.")

    # Check if the dataset has already been cloned to avoid redundant downloads
    if not os.path.exists("CREMA-D"):
        subprocess.run(["git", "clone", "https://github.com/CheyneyComputerScience/CREMA-D.git"], check=True)
        print("Repository cloned successfully.")
    else:
        print("Repository already cloned.")

    # Pull LFS files
    subprocess.run(["git", "lfs", "pull"], cwd="CREMA-D", check=True)
    print("Git LFS pull completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error while cloning the repository or pulling LFS files: {e}")
    sys.exit(1)

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
        f"None of the dataset paths {dataset_paths} exist. Please check the repository structure and try again."
    )

# Prepare a DataFrame for file paths and labels
audio_files = []
labels = []

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            # Assuming the filename format: ID_SentenceID_Emotion_Level.wav (e.g., 1001_DFA_ANG_XX.wav)
            parts = file.split('_')
            if len(parts) >= 4:
                label = parts[2]  # 'ANG' in the example
                audio_files.append(file_path)
                labels.append(label)
            else:
                print(f"Filename {file} does not match expected pattern.")

# Create a DataFrame
audio_df = pd.DataFrame({'file_path': audio_files, 'label': labels})
print(audio_df.head())

# Load the HuBERT feature extractor and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hubert_model.to(device)

# Function to extract features using HuBERT
def extract_features(file_path):
    try:
        audio_input, _ = librosa.load(file_path, sr=16000)
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input)

        # Process with feature extractor
        inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = hubert_model(**inputs)
        # Mean of last hidden states
        last_hidden_states = outputs.last_hidden_state
        feature = last_hidden_states.mean(dim=1).cpu().numpy().flatten()
        return feature
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        traceback.print_exc()
        return None

# Test feature extraction on one file
test_file = audio_df['file_path'].iloc[0]
print(f"Testing feature extraction on file: {test_file}")
test_feature = extract_features(test_file)
if test_feature is not None:
    print("Feature extraction successful.")
    print("Feature shape:", test_feature.shape)
else:
    print("Feature extraction failed.")
    sys.exit(1)

# Extract features for all files
features = []
new_labels = []
problematic_files = []

for index, row in audio_df.iterrows():
    feat = extract_features(row['file_path'])
    if feat is not None:
        features.append(feat)
        new_labels.append(row['label'])
    else:
        problematic_files.append(row['file_path'])

# Log problematic files
if problematic_files:
    with open("problematic_files.log", "w") as log_file:
        for problematic_file in problematic_files:
            log_file.write(f"{problematic_file}\n")
    print("Problematic files logged in problematic_files.log")

if not features:
    print("No features were extracted. Please check your audio files and paths.")
    sys.exit(1)

# Create DataFrame and encode labels
data = pd.DataFrame(features)
data['label'] = new_labels
encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])

X = data.iloc[:, :-1]
y = data['label']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(encoder.classes_))
y_test = to_categorical(y_test, num_classes=len(encoder.classes_))

# Build model
keras_model = Sequential()
keras_model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
keras_model.add(Dropout(0.5))
keras_model.add(Dense(128, activation='relu'))
keras_model.add(Dropout(0.5))
keras_model.add(Dense(64, activation='relu'))
keras_model.add(Dense(len(encoder.classes_), activation='softmax'))

keras_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
epochs = 50
batch_size = 32

history = keras_model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test)
)

# Evaluate the model
loss, accuracy = keras_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Generate additional evaluation metrics
y_pred = keras_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Classification Report
report = classification_report(y_true_classes, y_pred_classes, target_names=encoder.classes_)
print("Classification Report:")
print(report)

# Save the model
model_save_path = "emotion_recognition_using_Hubertmodel.h5"
keras_model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()