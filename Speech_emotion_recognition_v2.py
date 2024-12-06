import os
import subprocess
import sys
import torch
import librosa
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import traceback
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# OPTIONAL: Enable memory growth for GPU to help avoid memory fragmentation issues
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

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
        subprocess.run(["sudo", "apt-get", "install", "-y", "git-lfs"], check=True)
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

    # Check if the dataset has already been cloned
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

# Update the dataset path
dataset_paths = [
    "CREMA-D/AudioWAV",
    "CREMA-D/Audio",
    "CREMA-D/audio",
    "CREMA-D/AudioFiles"
]

dataset_path = None
for path in dataset_paths:
    if os.path.exists(path):
        dataset_path = path
        print(f"Using dataset path: {dataset_path}")
        break

if dataset_path is None:
    raise ValueError(
        f"None of the dataset paths {dataset_paths} exist. Check repository structure."
    )

# Prepare a DataFrame for file paths and labels
audio_files = []
labels = []
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            parts = file.split('_')
            if len(parts) >= 4:
                label = parts[2]
                audio_files.append(file_path)
                labels.append(label)
            else:
                print(f"Filename {file} does not match the expected pattern.")

audio_df = pd.DataFrame({'file_path': audio_files, 'label': labels})
print(audio_df.head())

# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Try an even smaller fixed_length
fixed_length = 50

def extract_features(file_path):
    try:
        audio_input, _ = librosa.load(file_path, sr=16000)
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input)
        inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state.cpu().numpy().squeeze()
        # Truncate or pad
        if last_hidden_states.shape[0] > fixed_length:
            last_hidden_states = last_hidden_states[:fixed_length, :]
        else:
            pad_len = fixed_length - last_hidden_states.shape[0]
            padding = np.zeros((pad_len, last_hidden_states.shape[1]))
            last_hidden_states = np.vstack((last_hidden_states, padding))
        return last_hidden_states
    except Exception as e:
        print(f"Error encountered for file: {file_path}")
        traceback.print_exc()
        return None

# Test feature extraction
test_file = audio_df['file_path'].iloc[0]
feature = extract_features(test_file)
if feature is not None:
    print("Feature extraction successful.")
    print("Feature shape:", feature.shape)
else:
    print("Feature extraction failed.")
    sys.exit(1)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(audio_df['label'])
num_classes = len(encoder.classes_)
# We'll keep y_encoded as integer labels and convert to categorical inside the generator

# Split into train/test
X_train_paths, X_test_paths, y_train_encoded, y_test_encoded = train_test_split(
    audio_df['file_path'], y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Data Generator
class DataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size, shuffle=True):
        self.file_paths = file_paths.tolist()
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_files = [self.file_paths[i] for i in batch_indexes]
        batch_labels = self.labels[batch_indexes]

        X_batch = []
        for fp in batch_files:
            feats = extract_features(fp)
            if feats is None:
                # If extraction fails, use a zero array as fallback
                feats = np.zeros((fixed_length, 768))
            X_batch.append(feats)

        X_batch = np.array(X_batch)
        y_batch = to_categorical(batch_labels, num_classes=num_classes)
        return X_batch, y_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Smaller batch_size
batch_size = 4

training_generator = DataGenerator(X_train_paths, y_train_encoded, batch_size=batch_size, shuffle=True)
validation_generator = DataGenerator(X_test_paths, y_test_encoded, batch_size=batch_size, shuffle=False)

sequence_length = fixed_length
feature_dim = 768  # from Wav2Vec2-base

# Simpler model
keras_model = Sequential()
keras_model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=(sequence_length, feature_dim)))
keras_model.add(Dropout(0.5))
keras_model.add(Bidirectional(LSTM(16)))
keras_model.add(Dropout(0.5))
keras_model.add(Dense(8, activation='relu'))
keras_model.add(Dense(num_classes, activation='softmax'))

keras_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

epochs = 20

history = keras_model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Evaluate
loss, accuracy = keras_model.evaluate(validation_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predictions
y_pred = keras_model.predict(validation_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test_encoded[:len(y_pred_classes)]

report = classification_report(y_true, y_pred_classes, target_names=encoder.classes_)
print("Classification Report:")
print(report)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Plot training history
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()