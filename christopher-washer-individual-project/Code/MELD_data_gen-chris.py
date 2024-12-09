import pandas as pd
import numpy as np
import os

train = pd.read_csv('MELD_train_sent_emo.csv')
test = pd.read_csv('MELD_test_sent_emo.csv')
dev = pd.read_csv('MELD_dev_sent_emo.csv')
crema = pd.read_csv('Crema_data.csv')

#print(train["Emotion"].unique())
print(np.sort(crema["Emotions"].unique()))

train = train[train['Emotion'] != 'surprise']
test = test[test['Emotion'] != 'surprise']
dev = dev[dev['Emotion'] != 'surprise']

crema_map = {'neutral': 'neutral', 'fear': 'fear', 'sadness': 'sad', 'joy': 'happy', 'disgust': 'disgust', 'anger': 'angry'}
train['Emotion'] = train['Emotion'].map(crema_map)
test['Emotion'] = test['Emotion'].map(crema_map)
dev['Emotion'] = dev['Emotion'].map(crema_map)

#print(np.sort(train["Emotion"].unique()))
#print(np.sort(test["Emotion"].unique()))
#print(np.sort(dev["Emotion"].unique()))

#train.to_csv('MELD_train_sent_emo_final.csv', index=False)
#test.to_csv('MELD_test_sent_emo_final.csv', index=False)
#dev.to_csv('MELD_dev_sent_emo_final.csv', index=False)

audio_directory = "/home/ubuntu/DL_Lectures/Final_Project/Data/MELD.Raw/train_audio"
# Create a new column for the file name
def generate_file_name(row):
    dialogue_id = f"dia{row['Dialogue_ID']}"
    utterance_id = f"utt{row['Utterance_ID']}"
    file_name = f"{dialogue_id}_{utterance_id}.wav"
    return os.path.join(audio_directory, file_name)

train['Path'] = train.apply(generate_file_name, axis=1)

# Check if the files exist in the directory

train['File_Exists'] = train['Path'].apply(lambda x: os.path.exists(x))
print(train['File_Exists'].value_counts())
train = train[train['File_Exists'] != False]
print(train['File_Exists'].value_counts())
train = train[['Emotion', 'Utterance', 'Path']]

# Save the updated DataFrame to a new CSV file
train.to_csv('MELD_train_sent_emo_final.csv', index=False)
#print(train.head())