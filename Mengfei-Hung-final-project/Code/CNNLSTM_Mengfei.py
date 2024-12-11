import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import IPython.display as ipd
from IPython.display import Audio
from pandas.core.interchange.dataframe_protocol import DataFrame
#from pydub import AudioSegment
#from pydub.playback import play
#from playsound import playsound
import librosa
#import pygame
import torch
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython.lib.display import Audio
from pygments.lexer import bygroups

#=========================================================================================================
# ---------------------------------------EDA--------------------------------------------------------
#=========================================================================================================
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f'Using {device}')
#path='C:/Users/feife/PycharmProjects/DL_Project/AudioWAV/'
path='/home/ubuntu/DL_Project/AudioWAV/'
data_dir_list = os.listdir(path)

paths = []
levels = []
labels = []
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        level = filename.split('_')[-1]
        level = level.split('.')[0]
        levels.append(level.lower())
        label = filename.split('_')[-2]
        label = label.split('_')[-1]
        labels.append(label.lower())
    if len(paths) == 10000:
        break
df = pd.DataFrame()
df['speech'] = paths
df['label']=labels
df['level']=levels
print(df.head())
print(df['speech'][0])


emotions_data = []

for it in df['speech']:
    # storing file emotions
    part = it.split('_')
    if part[3] == 'SAD':
        emotions_data.append('sad')
    elif part[3] == 'ANG':
        emotions_data.append('angry')
    elif part[3] == 'DIS':
        emotions_data.append('disgust')
    elif part[3] == 'FEA':
        emotions_data.append('fear')
    elif part[3] == 'HAP':
        emotions_data.append('happy')
    elif part[3] == 'NEU':
        emotions_data.append('neutral')
    else:
        emotions_data.append('Unknown')

level_data = []
for it in df['speech']:
    # storing file emotions
    part = it.split('.')[0].split('_')[4]
    if part == 'HI':
        level_data.append('High')
    elif part == 'LO':
        level_data.append('Low')
    elif part == 'MD':
        level_data.append('Medium')
    else:
        level_data.append('unspecified')

emotions_data_df = pd.DataFrame(emotions_data, columns=['Emotions'])
file_df = pd.DataFrame(data_dir_list, columns=['Files'])
level_df = pd.DataFrame(level_data, columns=['Emotion_Level'])
data=pd.concat([emotions_data_df,level_df,file_df], axis=1)

print(data.head())

print(data['Emotions'].value_counts())
print(data['Emotion_Level'].value_counts())
colors = sns.color_palette("husl", len(data['Emotions'].value_counts().index))
plt.bar(data['Emotions'].value_counts().index,data['Emotions'].value_counts().values, color=colors)
plt.show()

colors = sns.color_palette("husl", len(data['Emotion_Level'].value_counts().index))
plt.bar(data['Emotion_Level'].value_counts().index,data['Emotion_Level'].value_counts().values, color='green')
plt.show()

def emotions_count(emotions):
    emo = data[data['Emotions'] == emotions]
    print(emotions, emo['Emotion_Level'].value_counts())



emotions_count('angry')
emotions_count('disgust')
emotions_count('fear')
emotions_count('happy')
emotions_count('sad')
emotions_count('neutral')

'''
#=========================================================================================================
#-----------------------------------------playing audio ----------------------------------
#=========================================================================================================
pygame.init()
pygame.mixer.music.load('C:/Users/feife/PycharmProjects/DL_Project/AudioWAV/1001_DFA_ANG_XX.wav')
#pygame.mixer.music.play()




#=========================================================================================================
#--------------------Visualizing an audio waveform-------------------------
#=========================================================================================================
def visualizing_audio(file,text):
    signal, sr = librosa.load(path + file, sr=None)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(signal, sr=sr)
    plt.title(f'Waveform: {text}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

visualizing_audio('1001_DFA_ANG_XX.wav', 'Angry')
visualizing_audio('1001_DFA_DIS_XX.wav','Disgust')
visualizing_audio('1001_DFA_FEA_XX.wav','Fear')
visualizing_audio('1001_DFA_HAP_XX.wav','Happy')
visualizing_audio('1001_DFA_SAD_XX.wav','Sad')
visualizing_audio('1001_DFA_NEU_XX.wav','Neutral')
'''
#=========================================================================================================
#-------------------------------Data Augmentation-------------------------------------------
#=========================================================================================================

def augment_audio(y, sr):
    if np.random.rand() < 0.5:
        y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))
    if np.random.rand() < 0.5:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-3, 3))
    if np.random.rand() < 0.5:
        noise = 0.005 * np.random.randn(len(y))
        y = y + noise
    if np.random.rand() < 0.5:
        shift = int(sr * np.random.uniform(-0.2, 0.2))
        y = np.roll(y, shift)
    return y


#=========================================================================================================
#-------------------------------Extract Features-------------------------------------------
#=========================================================================================================
def extract_mfcc(filename):
    try:
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        #y = augment_audio(y, sr)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30).T, axis=0)
        return mfcc
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X_mfcc = [x for x in X_mfcc if x is not None]  # Remove None values
X = np.array(X_mfcc)
print(f"Extracted features shape: {X.shape}")



#=========================================================================================================
#-------------------------------Modeling-------------------------------------------
#=========================================================================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X = np.expand_dims(X, -1)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

# Convert to one-hot encoding
enc = OneHotEncoder(sparse_output=False)
y = enc.fit_transform(y.reshape(-1, 1))

print(f"y shape: {y.shape}")


# Dataset splitting (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Model parameters
input_size = X.shape[2]  # Number of features
hidden_size = 256  # Number of hidden units
num_layers = 3  # Number of LSTM layers
output_size = y.shape[1]  # Number of classes

# Define model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(CNNLSTMModel, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_prob)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Bidirectional LSTM doubles the hidden size

    def forward(self, x):
        # CNN forward pass
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, input_size, sequence_length)
        x = self.pool(F.relu(self.conv1(x)))  # Apply first convolution and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply second convolution and pooling

        # LSTM forward pass
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, sequence_length, features)
        out, _ = self.lstm(x)   # LSTM output
        out = self.dropout(out[:, -1, :])  # Take the last time step and apply dropout

        # Fully connected layer
        out = self.fc(out)
        return out

'''
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_prob, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.attention = nn.Linear(hidden_size * 2, 1)  # Attention layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Fully connected layer

    def forward(self, x):
        # LSTM output: (batch_size, sequence_length, hidden_size * 2)
        lstm_out, _ = self.lstm(x)

        # Compute attention weights
        attention_weights = torch.tanh(self.attention(lstm_out))  # (batch_size, sequence_length, 1)
        attention_weights = torch.softmax(attention_weights,
                                          dim=1)  # Normalize weights (batch_size, sequence_length, 1)

        # Weighted sum of LSTM outputs (apply attention weights)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size * 2)

        # Apply dropout and feed context vector into the final fully connected layer
        out = self.dropout(context_vector)
        out = self.fc(out)  # (batch_size, output_size)

        return out

'''
'''
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob,bidirectional=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)  # LSTM output: (batch_size, sequence_length, hidden_size)
        out = self.dropout(out[:, -1, :])  # Apply dropout
        out = self.fc(out)  # Fully connected layer
        return out
'''
#model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model = CNNLSTMModel(input_size, hidden_size, num_layers, output_size)
#model = LSTMWithAttention(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
batch_size = 32


# Function to create batches
def create_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]


train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, torch.argmax(y_batch, dim=1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == torch.argmax(y_batch, dim=1)).sum().item()
        total_predictions += y_batch.size(0)
    train_losses.append(train_loss / len(X_train))
    train_accuracy = correct_predictions / total_predictions
    train_accuracies.append(train_accuracy)



    # Calculate validation loss and accuracy
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0
        total = 0
        for X_batch, y_batch in create_batches(X_val, y_val, batch_size):
            outputs = model(X_batch)
            loss = criterion(outputs, torch.argmax(y_batch, dim=1))
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, actual = torch.max(y_batch, 1)
            val_correct += (predicted == actual).sum().item()
            total += y_batch.size(0)

        val_losses.append(val_loss / len(X_val))
        val_accuracy = val_correct / total
        val_accuracies.append(val_accuracy)

    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, torch.argmax(y_test, dim=1))
        _, test_predicted = torch.max(test_outputs, 1)
        _, test_actual = torch.max(y_test, 1)
        test_accuracy = (test_predicted == test_actual).float().mean()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {train_loss / len(X_train):.4f}, Train Accuracy: {train_accuracy:.4f} "
        f"Val Loss: {val_loss / len(X_val):.4f}, Val Accuracy: {val_accuracy:.4f}"
        f"Test Loss: {test_loss / len(X_test):.4f}, Test Accuracy: {test_accuracy:.4f}"
    )

# Plot train and validation accuracies
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy vs Epoch')
plt.legend()
plt.grid()
plt.show()

# Save the trained model
torch.save(model.state_dict(), "lstm_model.pth")
print("Model saved successfully as 'lstm_model.pth'.")



plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss vs Epoch')
plt.legend()
plt.grid()
plt.show()
print(model)

# Display model summary
#from torchinfo import summary
#dummy_input = torch.zeros((batch_size, X_train.shape[1], X_train.shape[2]))
#model_summary = summary(model, input_data=dummy_input, col_names=["input_size", "output_size", "num_params", "trainable"])
#print(model_summary)

