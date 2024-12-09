import pandas as pd
import os

OR_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd()
crema_dir = os.getcwd() + os.path.sep + 'Data' + os.path.sep + 'AudioWAV' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

crema_directory_list = os.listdir(crema_dir)

file_emotion = []
file_path = []

cremad_emotion_mapping = {
    "NEU": "Neutral",
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy/Joy",
    "SAD": "Sad"
}

cremad_sentence_mapping = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes"
}

for file in crema_directory_list:
    #Store file paths
    file_path.append(crema_dir + file)
    #store file emotions
    part = file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)
Crema_df.to_csv('Crema_data.csv', index=False)
print(Crema_df.head())

