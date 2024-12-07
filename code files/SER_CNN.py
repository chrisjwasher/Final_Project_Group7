#
# #-------------------------------------------------------------------------------------------------------------
# import os
# import pandas as pd
# import torchaudio
# import librosa
# import librosa.display
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
#
# # -------------------------------------------------------------------------------------------------------------
# # Step 1: Process CREMA-D and Generate `Crema_data.csv`
# def process_crema_d():
#     # Set the current directory
#     OR_PATH = os.getcwd()  # Current working directory
#     os.chdir("..")         # Go one level up
#
#     # Construct the path to the AudioWAV directory
#     PATH = os.getcwd()     # Parent directory
#     crema_dir = os.path.join(PATH, 'CREMA-D', 'AudioWAV')
#
#     # Return to the original directory after setting up paths
#     os.chdir(OR_PATH)
#
#     # List all files in the AudioWAV directory
#     crema_directory_list = os.listdir(crema_dir)
#
#     # Initialize lists to store file paths and emotions
#     file_emotion = []
#     file_path = []
#
#     # Process each file in the directory
#     for file in crema_directory_list:
#         if file.endswith(".wav"):  # Ensure we're processing .wav files
#             # Store file paths
#             file_path.append(os.path.join(crema_dir, file))
#             # Store file emotions
#             part = file.split('_')
#             if len(part) > 2:  # Ensure the file name has enough parts
#                 if part[2] == 'SAD':
#                     file_emotion.append('sad')
#                 elif part[2] == 'ANG':
#                     file_emotion.append('angry')
#                 elif part[2] == 'DIS':
#                     file_emotion.append('disgust')
#                 elif part[2] == 'FEA':
#                     file_emotion.append('fear')
#                 elif part[2] == 'HAP':
#                     file_emotion.append('happy')
#                 elif part[2] == 'NEU':
#                     file_emotion.append('neutral')
#                 else:
#                     file_emotion.append('Unknown')
#
#     # Create DataFrames
#     emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
#     path_df = pd.DataFrame(file_path, columns=['Path'])
#
#     # Combine the DataFrames
#     Crema_df = pd.concat([emotion_df, path_df], axis=1)
#
#     # Save to CSV
#     Crema_df.to_csv('Crema_data.csv', index=False)
#
#     # Print a sample of the dataframe
#     print(Crema_df.head())
#
#     return Crema_df
#
# # -------------------------------------------------------------------------------------------------------------
# # Step 2: Extract Mel Spectrograms
# def extract_mel_spectrogram(audio_path, n_mels=64, max_len=128):
#     y, sr = librosa.load(audio_path, sr=16000)  # Load audio, resample to 16kHz
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB
#     # Pad or truncate to ensure consistent size
#     if mel_spec_db.shape[1] < max_len:
#         pad_width = max_len - mel_spec_db.shape[1]
#         mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mel_spec_db = mel_spec_db[:, :max_len]
#     return mel_spec_db
#
# # -------------------------------------------------------------------------------------------------------------
# # Step 3: Train-Test Split and Feature Extraction
# def prepare_data(dataframe):
#     train_df, test_df = train_test_split(dataframe, test_size=0.2, stratify=dataframe['Emotions'], random_state=42)
#
#     X_train, y_train = [], []
#     X_test, y_test = [], []
#
#     label_map = {emotion: idx for idx, emotion in enumerate(dataframe['Emotions'].unique())}
#
#     for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing Train Data"):
#         mel_spec = extract_mel_spectrogram(row['Path'])
#         X_train.append(mel_spec)
#         y_train.append(label_map[row['Emotions']])
#
#     for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Test Data"):
#         mel_spec = extract_mel_spectrogram(row['Path'])
#         X_test.append(mel_spec)
#         y_test.append(label_map[row['Emotions']])
#
#     X_train = np.array(X_train).reshape(-1, 64, 128, 1)
#     X_test = np.array(X_test).reshape(-1, 64, 128, 1)
#     y_train = to_categorical(y_train, num_classes=len(label_map))
#     y_test = to_categorical(y_test, num_classes=len(label_map))
#
#     return X_train, X_test, y_train, y_test, label_map
#
# # -------------------------------------------------------------------------------------------------------------
# # Step 4: Define CNN Model
# def create_cnn_model(input_shape, num_classes):
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         MaxPooling2D((2, 2)),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')  # Number of emotion classes
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
#
# # -------------------------------------------------------------------------------------------------------------
# # Step 5: Train the Model
# def train_cnn_model(X_train, X_test, y_train, y_test, input_shape, num_classes):
#     model = create_cnn_model(input_shape, num_classes)
#     model.summary()
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)
#     return model
#
# # -------------------------------------------------------------------------------------------------------------
# # Entry point for the script
# if __name__ == "__main__":
#     # Step 1: Process CREMA-D data
#     crema_df = process_crema_d()
#
#     # Step 2: Prepare data for CNN
#     X_train, X_test, y_train, y_test, label_map = prepare_data(crema_df)
#
#     # Step 3: Train CNN model
#     input_shape = (64, 128, 1)  # Shape of Mel spectrograms
#     num_classes = len(label_map)
#     model = train_cnn_model(X_train, X_test, y_train, y_test, input_shape, num_classes)


# import os
# import pandas as pd
# import librosa
# import numpy as np
# from tqdm import tqdm
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
#
# # -------------------------------------------------------------------------------------------------------------
# # Step 1: Generate `Crema_data.csv`
# def generate_crema_data_csv():
#     """
#     Generates Crema_data.csv with file paths and emotion labels from the CREMA-D dataset.
#     """
#     OR_PATH = os.getcwd()  # Current working directory
#     os.chdir("..")         # Go one level up
#     PATH = os.getcwd()     # Parent directory
#     crema_dir = os.path.join(PATH,'DL_Project', 'CREMA-D', 'AudioWAV') + os.path.sep
#     os.chdir(OR_PATH)      # Return to original directory
#
#     crema_directory_list = os.listdir(crema_dir)
#
#     file_emotion = []
#     file_path = []
#
#     for file in crema_directory_list:
#         # Store file paths
#         file_path.append(crema_dir + file)
#         # Store file emotions
#         part = file.split('_')
#         if part[2] == 'SAD':
#             file_emotion.append('sad')
#         elif part[2] == 'ANG':
#             file_emotion.append('angry')
#         elif part[2] == 'DIS':
#             file_emotion.append('disgust')
#         elif part[2] == 'FEA':
#             file_emotion.append('fear')
#         elif part[2] == 'HAP':
#             file_emotion.append('happy')
#         elif part[2] == 'NEU':
#             file_emotion.append('neutral')
#         else:
#             file_emotion.append('Unknown')
#
#     # DataFrame for emotion of files
#     emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
#
#     # DataFrame for path of files
#     path_df = pd.DataFrame(file_path, columns=['Path'])
#     Crema_df = pd.concat([emotion_df, path_df], axis=1)
#
#     # Save to CSV
#     Crema_df.to_csv('Crema_data.csv', index=False)
#     print(f"Generated Crema_data.csv with {len(Crema_df)} entries.")
#     print(Crema_df.head())
#
#     return Crema_df
#
# # -------------------------------------------------------------------------------------------------------------
# # Step 2: Extract Mel Spectrograms
# def extract_mel_spectrogram(audio_path, n_mels=64, max_len=128):
#     """
#     Extracts Mel spectrogram features from an audio file.
#     """
#     try:
#         y, sr = librosa.load(audio_path, sr=16000)  # Load audio, resample to 16kHz
#         mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
#         mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB
#         # Pad or truncate to ensure consistent size
#         if mel_spec_db.shape[1] < max_len:
#             pad_width = max_len - mel_spec_db.shape[1]
#             mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
#         else:
#             mel_spec_db = mel_spec_db[:, :max_len]
#         return mel_spec_db
#     except Exception as e:
#         print(f"Error processing file {audio_path}: {e}")
#         return None
#
# # -------------------------------------------------------------------------------------------------------------
# # Step 3: Train-Test Split and Feature Extraction
# def prepare_data(dataframe):
#     """
#     Prepares the data by extracting Mel spectrograms and splitting into train and test sets.
#     """
#     train_df, test_df = train_test_split(dataframe, test_size=0.2, stratify=dataframe['Emotions'], random_state=42)
#
#     X_train, y_train = [], []
#     X_test, y_test = [], []
#
#     label_map = {emotion: idx for idx, emotion in enumerate(dataframe['Emotions'].unique())}
#
#     for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing Train Data"):
#         mel_spec = extract_mel_spectrogram(row['Path'])
#         if mel_spec is not None:
#             X_train.append(mel_spec)
#             y_train.append(label_map[row['Emotions']])
#
#     for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Test Data"):
#         mel_spec = extract_mel_spectrogram(row['Path'])
#         if mel_spec is not None:
#             X_test.append(mel_spec)
#             y_test.append(label_map[row['Emotions']])
#
#     X_train = np.array(X_train).reshape(-1, 64, 128, 1)
#     X_test = np.array(X_test).reshape(-1, 64, 128, 1)
#     y_train = to_categorical(y_train, num_classes=len(label_map))
#     y_test = to_categorical(y_test, num_classes=len(label_map))
#
#     return X_train, X_test, y_train, y_test, label_map
#
# # -------------------------------------------------------------------------------------------------------------
# # Step 4: Define CNN Model
# def create_cnn_model(input_shape, num_classes):
#     """
#     Defines the CNN architecture.
#     """
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         MaxPooling2D((2, 2)),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')  # Number of emotion classes
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
#
# # -------------------------------------------------------------------------------------------------------------
# # Step 5: Train the Model
# def train_cnn_model(X_train, X_test, y_train, y_test, input_shape, num_classes):
#     """
#     Trains the CNN model on the extracted features.
#     """
#     model = create_cnn_model(input_shape, num_classes)
#     model.summary()
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)
#     model.save("emotion_cnn_model.h5")
#     print("Model saved as emotion_cnn_model.h5")
#     return model
#
# # -------------------------------------------------------------------------------------------------------------
# # Entry point for the script
# if __name__ == "__main__":
#     # Step 1: Generate Crema_data.csv
#     crema_df = generate_crema_data_csv()
#
#     # Step 2: Prepare data for CNN
#     X_train, X_test, y_train, y_test, label_map = prepare_data(crema_df)
#
#     # Step 3: Train CNN model
#     input_shape = (64, 128, 1)  # Shape of Mel spectrograms
#     num_classes = len(label_map)
#     model = train_cnn_model(X_train, X_test, y_train, y_test, input_shape, num_classes)
#
# # Evaluate the model on the test data
# test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


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
# Step 3: Train-Test Split and Feature Extraction
def prepare_data(dataframe):
    train_df, test_df = train_test_split(dataframe, test_size=0.2, stratify=dataframe['Emotions'], random_state=42)

    X_train, y_train, X_test, y_test = [], [], [], []

    label_map = {emotion: idx for idx, emotion in enumerate(dataframe['Emotions'].unique())}

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing Train Data"):
        mel_spec = extract_mel_spectrogram(row['Path'])
        if mel_spec is not None:
            X_train.append(mel_spec)
            y_train.append(label_map[row['Emotions']])

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Test Data"):
        mel_spec = extract_mel_spectrogram(row['Path'])
        if mel_spec is not None:
            X_test.append(mel_spec)
            y_test.append(label_map[row['Emotions']])

    X_train = np.array(X_train).reshape(-1, 64, 128, 1)
    X_test = np.array(X_test).reshape(-1, 64, 128, 1)
    y_train = to_categorical(y_train, num_classes=len(label_map))
    y_test = to_categorical(y_test, num_classes=len(label_map))

    return X_train, X_test, y_train, y_test, label_map


# -------------------------------------------------------------------------------------------------------------
# Step 4: Define Enhanced CNN Model
def create_enhanced_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# -------------------------------------------------------------------------------------------------------------
# Step 5: Train the Enhanced Model
def train_and_evaluate_model(X_train, X_test, y_train, y_test, input_shape, num_classes):
    model = create_enhanced_cnn_model(input_shape, num_classes)
    model.summary()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=16)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plot Accuracy and Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

    # Confusion Matrix
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_map.keys()),
                yticklabels=list(label_map.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return model


# -------------------------------------------------------------------------------------------------------------
# Entry point for the script
if __name__ == "__main__":
    crema_df = generate_crema_data_csv()
    X_train, X_test, y_train, y_test, label_map = prepare_data(crema_df)
    input_shape = (64, 128, 1)
    num_classes = len(label_map)
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test, input_shape, num_classes)
