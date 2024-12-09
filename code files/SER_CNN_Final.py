# -------------------------------------------------------------------------------------------------------------
# importing libraries
import os
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------------------------------------------------------
# Step 1: Generate `Crema_data.csv`
def generate_crema_data_csv():
    """
    Generates Crema_data.csv with file paths and emotion labels from the CREMA-D dataset.
    """
    OR_PATH = os.getcwd()
    os.chdir("..")
    PATH = os.getcwd()
    crema_dir = os.path.join(PATH, 'DL_Project', 'CREMA-D', 'AudioWAV') + os.path.sep
    os.chdir(OR_PATH)

    crema_directory_list = os.listdir(crema_dir)

    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        try:
            file_path.append(crema_dir + file)
            part = file.split('_')
            emotion_map = {'SAD': 'sad', 'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy',
                           'NEU': 'neutral'}
            file_emotion.append(emotion_map.get(part[2], 'Unknown'))
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    crema_df = pd.concat([emotion_df, path_df], axis=1)
    crema_df.to_csv('Crema_data.csv', index=False)

    print(f"Generated Crema_data.csv with {len(crema_df)} entries.")
    return crema_df


# -------------------------------------------------------------------------------------------------------------
# Step 2: Extract Mel Spectrograms
def extract_mel_spectrogram(audio_path, n_mels=64, max_len=128):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] < max_len:
            pad_width = max_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_len]
        return mel_spec_db
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return None


# -------------------------------------------------------------------------------------------------------------
# Step 3: Data Augmentation



def time_shift(audio, shift_max=0.2):
    """
    Apply a random time shift to the audio.
    """
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)

def add_noise(audio, noise_level=0.005):
    """
    Add random noise to the audio.
    """
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise


def time_stretch(audio, rate, sr=16000):
    """
    Stretch or compress the audio in time.

    Parameters:
    - audio (np.ndarray): The audio signal (1D waveform).
    - rate (float): The stretch factor. Values >1.0 speed up, <1.0 slow down.
    - sr (int): Sample rate (default=16000).

    Returns:
    - np.ndarray: The time-stretched audio signal.
    """
    if audio.ndim > 1:  # If stereo, convert to mono
        audio = librosa.to_mono(audio)

    if len(audio) < sr * 0.5:  # Ensure at least 0.5 seconds of audio
        raise ValueError(f"Audio signal too short for time stretch: {len(audio)} samples")

    # Convert waveform to spectrogram
    stft = librosa.stft(audio)
    stretched_stft = librosa.effects.time_stretch(stft, rate)

    # Convert spectrogram back to waveform
    stretched_audio = librosa.istft(stretched_stft)
    return stretched_audio



def pitch_shift(audio, sr, n_steps):
    """
    Shift the pitch of an audio signal.

    Parameters:
    - audio (np.ndarray): The audio signal.
    - sr (int): The sample rate.
    - n_steps (int): Number of half-steps to shift.

    Returns:
    - np.ndarray: The pitch-shifted audio signal.
    """
    if len(audio.shape) > 1:  # If stereo, convert to mono
        audio = librosa.to_mono(audio)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def augment_audio(audio, sr):
    """Apply augmentations to the audio."""

    # Convert stereo to mono if necessary
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    # Validate audio length
    if len(audio) < 2:
        print(f"Audio too short for augmentation, skipping: length={len(audio)}")
        return audio

    # Augmentations
    # if np.random.rand() < 0.5:  # Time stretching removed
    #     audio = time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    if np.random.rand() < 0.5:
        audio = pitch_shift(audio, sr, n_steps=np.random.randint(-2, 3))
    if np.random.rand() < 0.5:
        audio = add_noise(audio)
    if np.random.rand() < 0.5:
        audio = time_shift(audio)

    return audio

# -------------------------------------------------------------------------------------------------------------
# Step 4: Train-Test Split and Feature Extraction
def prepare_data(dataframe, augment=False):
    train_df, test_df = train_test_split(dataframe, test_size=0.2, stratify=dataframe['Emotions'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['Emotions'], random_state=42)

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    label_map = {emotion: idx for idx, emotion in enumerate(dataframe['Emotions'].unique())}

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing Train Data"):
        y, sr = librosa.load(row['Path'], sr=16000)
        if augment:
            y = augment_audio(y, sr)
        mel_spec = extract_mel_spectrogram(row['Path'])
        if mel_spec is not None:
            X_train.append(mel_spec)
            y_train.append(label_map[row['Emotions']])

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Processing Validation Data"):
        mel_spec = extract_mel_spectrogram(row['Path'])
        if mel_spec is not None:
            X_val.append(mel_spec)
            y_val.append(label_map[row['Emotions']])

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Test Data"):
        mel_spec = extract_mel_spectrogram(row['Path'])
        if mel_spec is not None:
            X_test.append(mel_spec)
            y_test.append(label_map[row['Emotions']])

    X_train = np.array(X_train).reshape(-1, 64, 128, 1)
    X_val = np.array(X_val).reshape(-1, 64, 128, 1)
    X_test = np.array(X_test).reshape(-1, 64, 128, 1)
    y_train = to_categorical(y_train, num_classes=len(label_map))
    y_val = to_categorical(y_val, num_classes=len(label_map))
    y_test = to_categorical(y_test, num_classes=len(label_map))

    return X_train, X_val, X_test, y_train, y_val, y_test, label_map
# -------------------------------------------------------------------------------------------------------------
# Step 5: Define Enhanced CNN Model

def create_enhanced_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # Output: 32x64x32

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # Output: 16x32x64

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # Output: 8x16x128

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # Output: 4x8x256

        # Block 5
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # Output: 2x4x512

        # Block 6 (Last Pooling Removed)
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),

        # Flatten or Global Average Pooling
        Flatten(),

        # Fully connected layers
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# -------------------------------------------------------------------------------------------------------------
# Step 5: Train and Evaluate Enhanced Model

def train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, input_shape, num_classes):
    model = create_enhanced_cnn_model(input_shape, num_classes)
    model.summary()

    # Train the model with validation data
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16)

    # Save the model
    model.save("emotion_recognition_model.h5")
    print("Model saved as emotion_recognition_model.h5")

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # ---------------------------- Visualizations ----------------------------
    # Accuracy and Loss plots
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_plots.png")  # Save the plot
    print("Training plots saved as training_plots.png")

    # ---------------------------- Confusion Matrix ----------------------------
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_map.keys()),
                yticklabels=list(label_map.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("confusion_matrix.png")  # Save the confusion matrix
    print("Confusion matrix saved as confusion_matrix.png")

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=list(label_map.keys()), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report.csv", index=True)
    print("Classification report saved as classification_report.csv")

    return model

# -------------------------------------------------------------------------------------------------------------
# Entry point for the script

if __name__ == "__main__":
    crema_df = generate_crema_data_csv()

    # Enable data augmentation during data preparation
    X_train, X_val, X_test, y_train, y_val, y_test, label_map = prepare_data(crema_df, augment=True)

    input_shape = (64, 128, 1)
    num_classes = len(label_map)

    # Train and evaluate the model
    model = train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, input_shape, num_classes)
