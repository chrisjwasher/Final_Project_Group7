import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import plotly.graph_objects as go
from PIL import Image

import IPython.display as ipd
from IPython.display import Audio
from pandas.core.interchange.dataframe_protocol import DataFrame
from pydub import AudioSegment
from pydub.playback import play
from playsound import playsound
import librosa
import pygame
import torch
import os
import sys
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from IPython.lib.display import Audio
from pygments.lexer import bygroups
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from tensorflow.keras.models import load_model


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f'Using {device}')
path='C:/Users/feife/PycharmProjects/pythonProject/AudioWAV/'
#path='/home/ubuntu/DL_Project/AudioWAV/'
data_dir_list = os.listdir(path)




# =========================================================================================================
#                                   Custom CSS
# =========================================================================================================
alt.themes.enable('dark')
st.set_page_config(
    page_title="Campaign Speech Emotion Recognition",
    page_icon=':derelict_house_building:',
    layout="wide",
    initial_sidebar_state="expanded")

background_style = """
<style>
    /* Set background color for the main content */
    .stApp {
        background-color:#000000; /* Choose your preferred color here #FFDAB9 */
	color: white; 
    }

        /* Increase font size and adjust DataFrame styling */
    .dataframe, .dataframe th, .dataframe td {
        font-size: 30px; /* Adjust this to your preferred font size */
        #background-color: black; /* Set a background color if desired */
        padding: 10px; /* Increase padding for better spacing */
</style>
"""

# Apply the CSS style
st.markdown(background_style, unsafe_allow_html=True)

st.markdown("<h1 style= 'text-align:center; color:white;  '>Campaign Speech Emotion Recognition</h1> ",
            unsafe_allow_html=True)
st.write('')
#st.markdown('Data Source: ')

selectbox_style = """
	<style>
	.stSelectbox >div[data-baseweb="select"]> div {height: 180% !important;
	      padding: 5px; font-family: 'Arial' !important; border: 2px solid #be0051 !important; font-weight: bold; 
	      font-size: 20px; 
	}
	</style>
	"""
st.markdown(selectbox_style, unsafe_allow_html=True)

selectnumber_style = """
	<style>
	.stNumberInput >div > input {
        height: 180% !important;
        padding: 5px;
        font-family: 'Arial' !important;
        border: 2px solid #be0051 !important;
        font-weight: bold;
        font-size: 20px;
    }
	</style>
	"""
st.markdown(selectnumber_style, unsafe_allow_html=True)

sidebar_style = """
<style>
    /* Set sidebar background color */
    .stSidebar {
        background-color: #007C80 !important; /* Sidebar background color */
        color: #FAFAFA !important;
    }
    /* Ensure all text in the sidebar is white */
    .stSidebar, .stSidebar * {
        color: #FAFAFA !important; 
        font-size: 25px; /* Set all text in sidebar to white */
    }
    /* Specifically target radio button labels */
    div[role="radiogroup"] label {
        color: #FAFAFA !important; 
        font-size: 44px !important; 
        font-weight: bold; /* Force radio button text to white */
    }
    div[role="radiogroup"] {
        font-size: 35px !important; 
        padding: 5px; 
        font-weight: bold; /* This affects the entire radio group, including labels */
    }
</style>
"""

st.markdown(sidebar_style, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h1 style= 'color:white; font-size: 35px '> Select a Sector </h1>", unsafe_allow_html=True)#FAFAFA
    # data_viz_button = st.sidebar.button("Data Visualization",use_container_width=False,icon="🚨",on_click=callable,)
    # prediction_button = st.sidebar.button("Prediction",use_container_width=False,icon="🚨",on_click=callable,)

    page = st.sidebar.radio('', ["📊Model", "📈Demonstration"])


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
    if part[2] == 'SAD':
        emotions_data.append('sad')
    elif part[2] == 'ANG':
        emotions_data.append('angry')
    elif part[2] == 'DIS':
        emotions_data.append('disgust')
    elif part[2] == 'FEA':
        emotions_data.append('fear')
    elif part[2] == 'HAP':
        emotions_data.append('happy')
    elif part[2] == 'NEU':
        emotions_data.append('neutral')
    else:
        emotions_data.append('Unknown')

level_data = []
for it in df['speech']:
    # storing file emotions
    part = it.split('.')[0].split('_')[3]
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
print(data)



# =========================================================================================================
#                                   EDA
# =========================================================================================================
# st.sidebar.title('Choose Sector')
# page=st.sidebar.radio(['','Data Visualization', 'Prediction'])

# ---------------------------------------EDA--------------------------------------------------------

if page == '📊Model':
    st.title('Data')
    st.dataframe(data,height=300)

    st.title('EDA')
    c1, c2 = st.columns((1,1), gap='medium')
    with c1:
        st.markdown('Emotion Type Distribution')
        colors = sns.color_palette("husl", len(data['Emotions'].value_counts().index))
        #plt.bar(data['Emotions'].value_counts().index, data['Emotions'].value_counts().values, color=colors)
        fig = px.bar(data, x=data['Emotions'].value_counts().index, y=data['Emotions'].value_counts().values, color=data['Emotions'].value_counts().index)
        fig.update_layout(
            title_text='Emotion distribution',
            xaxis_title='Emotion type',
            yaxis_title='count',
            bargap=0.2,
            width=400,
            height=350
        )
        st.plotly_chart(fig)
        label1 = data['Emotions'].value_counts().index
        level1 = data['Emotions'].value_counts().values
        data1 = pd.DataFrame({'Emotions': label1, 'Count': level1})
        st.dataframe(data1, use_container_width=True)

    with c2:
        st.markdown('Emotion Level Distribution')
        fig = px.bar(data, x=data['Emotion_Level'].value_counts().index, y=data['Emotion_Level'].value_counts().values)
        fig.update_layout(
            title_text='Emotion level distribution',
            xaxis_title='Emotion level',
            yaxis_title='count',
            bargap=0.2,
            width=400,
            height=350
        )
        st.plotly_chart(fig)
        label2=data['Emotion_Level'].value_counts().index
        level2=data['Emotion_Level'].value_counts().values
        data2=pd.DataFrame({'Emotion Level':label2,'Count':level2})
        st.dataframe(data2, use_container_width=True)


    def emotions_count(emotions):
        emo = data[data['Emotions'] == emotions]
        counts = emo['Emotion_Level'].value_counts()
        return counts.reset_index().rename(columns={'index': 'emotion', 'Emotion_Level': f'{emotions}'})


    st.markdown('Emotion Type and Emotion Level')
    def emo_type_level_bar(emotion):
        fig = px.bar(emotions_count(emotion), x=emotion, y='count')
        fig.update_layout(
            title_text=emotion,
            xaxis_title='Emotion level',
            yaxis_title='count',
            bargap=0.2,
            width=200,
            height=200
        )
        st.plotly_chart(fig)

    c1,c2, c3=st.columns((1,1,1),gap='medium')
    with c1:
        emo_type_level_bar('angry')
        emo_type_level_bar('disgust')

        #st.dataframe(emotions_count('angry'), use_container_width=True, height=150)
        #st.dataframe(emotions_count('disgust'), use_container_width=True, height=150)

    with c2:
        emo_type_level_bar('fear')
        emo_type_level_bar('happy')


    with c3:
        emo_type_level_bar('sad')
        emo_type_level_bar('neutral')





#-----------------------------------------playing audio ----------------------------------

    st.markdown('Sound Display and Sound Waveform')
    audio_path1 = path+'1001_DFA_ANG_XX.wav'
    audio_path2 = path+'1001_DFA_DIS_XX.wav'
    audio_path3 = path+'1001_DFA_FEA_XX.wav'
    audio_path4 = path+'1001_DFA_HAP_XX.wav'
    audio_path5 = path+'1001_DFA_SAD_XX.wav'
    audio_path6 = path+'1001_DFA_NEU_XX.wav'




    def visualizing_audio(file, text):
        signal, sr = librosa.load(path + file, sr=None)
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(signal, sr=sr,color='purple')
        plt.title(f'Waveform: {text}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        st.pyplot(plt)


    c1,c2= st.columns((1,1), gap='medium')
    with c1:
        st.write('Angry')
        visualizing_audio('1001_DFA_ANG_XX.wav', 'Angry')
        st.audio(audio_path1, format='audio/wav')
        st.write('Disgust')
        visualizing_audio('1001_DFA_DIS_XX.wav', 'Disgust')
        st.audio(audio_path2, format='audio/wav')
        st.write('Fear')
        visualizing_audio('1001_DFA_FEA_XX.wav', 'Fear')
        st.audio(audio_path3, format='audio/wav')
    with c2:
        st.write('Happy')
        visualizing_audio('1001_DFA_HAP_XX.wav', 'Happy')
        st.audio(audio_path4, format='audio/wav')
        st.write('Sad')
        visualizing_audio('1001_DFA_SAD_XX.wav', 'Sad')
        st.audio(audio_path5, format='audio/wav')
        st.write('Neutral')
        visualizing_audio('1001_DFA_NEU_XX.wav', 'Neutral')
        st.audio(audio_path6, format='audio/wav')


    st.title('Model')
    st.subheader('CNN Model')

    c1, c2 = st.columns((1, 1), gap='medium')
    with c1:
        st.image("CNN_process.jpg", caption="Process")
        #st.image("CNN_architecture.jpg", caption="Model Architecture")
    with c2:
        st.image("CNN_accuracy.jpg", caption="Model Accuracy and Loss")
        st.image("CNN_classification.jpg", caption="Classification Results")


    st.subheader('CNNLSTM Model')
    c1, c2 = st.columns((1, 1), gap='medium')
    with c1:
        st.image("CNNLSTM_process.jpg", caption="Process")
        st.image("CNNLSTM_architecture.jpg", caption="Model Architecture",width=410)

    with c2:
        st.image("CNNLSTM_accuracy.jpg", caption="Model Accuracy")
        st.image("CNNLSTM_loss.jpg", caption="Loss")


    st.subheader('Pre-trained Model')
    c1, c2 = st.columns((1, 1), gap='medium')
    with c1:
        st.image("HUBERT_architecture.jpg", caption="Model Architecture")
        #st.image("sunrise.jpg", caption="Sunrise by the mountains")
    with c2:
        st.image("HUBERT_accuracy.png", caption="Model Accuracy")
        st.image("HUBERT_loss.png", caption="Model Loss")

    st.subheader('AST Model')
    c1, c2 = st.columns((1, 1), gap='medium')
    with c1:
    #    st.image("sunrise.jpg", caption="Sunrise by the mountains")
        st.image("AST_architecture.jpg", caption="Model Architecture")
    with c2:
        st.image("AST_accuracy.jpg", caption="Model Accuracy")
    #    st.image("sunrise.jpg", caption="Sunrise by the mountains")
        
    st.subheader('Multi Model')
    c1, c2 = st.columns((1, 1), gap='medium')
    with c1:
        st.image("Multi_architecture.jpg", caption="Model Architecture")
    #    st.image("sunrise.jpg", caption="Sunrise by the mountains")
    with c2:
        st.image("Multi_accuracy.jpg", caption="Model Accuracy")
    #    st.image("sunrise.jpg", caption="Sunrise by the mountains")




# ------------------------------------------Prediction------------------------------------------

elif page == '📈Demonstration':

    # %%
    import pandas as pd
    import joblib
    from tensorflow.keras.models import load_model
    import traceback
    import warnings

    # List of audio file paths to test
    audio_file_paths = [
        "C:/Users/feife/PycharmProjects/DL_Project/AudioTest/Highlights_debate_emotion_clip_1.mp3",
        "C:/Users/feife/PycharmProjects/DL_Project/AudioTest/Highlights_debate_emotion_clip_2.mp3",
        "C:/Users/feife/PycharmProjects/DL_Project/AudioTest/Highlights_debate_emotion_clip_3.mp3",
        "C:/Users/feife/PycharmProjects/DL_Project/AudioTest/Highlights_debate_emotion_clip_4.mp3",
        "C:/Users/feife/PycharmProjects/DL_Project/AudioTest/Highlights_debate_emotion_clip_5.mp3",
        "C:/Users/feife/PycharmProjects/DL_Project/AudioTest/Highlights_debate_emotion_clip_6.mp3",
        "C:/Users/feife/PycharmProjects/DL_Project/AudioTest/Highlights_debate_emotion_clip_7.mp3",
        "C:/Users/feife/PycharmProjects/DL_Project/AudioTest/Highlights_debate_emotion_clip_8.mp3",
        "C:/Users/feife/PycharmProjects/DL_Project/AudioTest/Highlights_debate_emotion_clip_9.mp3",
        "C:/Users/feife/PycharmProjects/DL_Project/AudioTest/Highlights_debate_emotion_clip_10.mp3"]

    # Load HuBERT
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hubert_model.to(device)


    def extract_features(file_path):
        try:
            # Load audio
            audio_input, _ = librosa.load(file_path, sr=16000)
            if not isinstance(audio_input, np.ndarray):
                audio_input = np.array(audio_input)

            # Extract features using HuBERT
            inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = hubert_model(**inputs)
            # Average last hidden states over time
            last_hidden_states = outputs.last_hidden_state
            feature = last_hidden_states.mean(dim=1).cpu().numpy().flatten()
            return feature
        except Exception as e:
            print(f"Error encountered while parsing file: {file_path}")
            traceback.print_exc()
            return None


    # Load the pre-trained classification model
    model_path = "emotion_recognition_using_Hubertmodel.h5"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Make sure it's in the current directory.")
        sys.exit(1)

    keras_model = load_model(model_path)

    # The classes should be the same as used during training
    classes = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    encoder = LabelEncoder()
    encoder.fit(classes)

    st.title("Emotion Recognition from Speech")
    selected_file = st.selectbox("Select an audio file to process:", audio_file_paths)
    st.audio(selected_file)
    if st.button("Predict Emotion"):
        st.spinner("Processing...")
        #st.write(f"Processing file: {selected_file}")
        test_feature = extract_features(selected_file)
        if test_feature is None:
            st.error(f"Failed to extract features for {selected_file}.")
        else:
            X_test = np.array([test_feature])
            y_pred = keras_model.predict(X_test)
            y_pred_class = np.argmax(y_pred, axis=1)
            predicted_emotion = encoder.inverse_transform(y_pred_class)[0]
            st.success(f"Predicted Emotion: {predicted_emotion}")


#    model_path = "emotion_recognition_model.h5"
#    emotion_model = load_model(model_path)
#    print(f"Loaded trained model from {model_path}")

    # Load the label encoder
#    encoder_path = "label_encoder.joblib"
#    encoder = joblib.load(encoder_path)
#    print("Loaded label encoder.")

    # Load Wav2Vec2 processor and model
#    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
#    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    # Move the model to GPU if available
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    wav2vec_model.to(device)


    # Function to extract features
#    def extract_features(file_path):
#        try:
#            audio_input, _ = librosa.load(file_path, sr=16000)
#            if not isinstance(audio_input, np.ndarray):
#                audio_input = np.array(audio_input)
#            inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
#            inputs = {key: val.to(device) for key, val in inputs.items()}
#            with torch.no_grad():
#                outputs = wav2vec_model(**inputs)
            # Mean of last hidden states
#            last_hidden_states = outputs.last_hidden_state
    #            feature = last_hidden_states.mean(dim=1).cpu().numpy().flatten()
#            return feature
#        except Exception as e:
#            print(f"Error encountered while processing file: {file_path}")
#            traceback.print_exc()
#            return None


    # Function to predict emotion
    def predict_emotion(file_path):
        features = extract_features(file_path)
        if features is None:
            print("Failed to extract features. Please check the audio file.")
            return None

        # Predict using the model
        predictions = keras_model.predict(np.array([features]))
        predicted_class = np.argmax(predictions, axis=1)

        # Decode the label
        emotion_label = encoder.inverse_transform(predicted_class)
        print(f"Predicted Emotion: {emotion_label[0]}")
        return emotion_label[0]


    st.title("Emotion Recognition from Speech")
    st.write("Upload an audio file to predict the emotion.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav","mp3"])

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            # Save the uploaded file temporarily
            temp_file = "temp_audio_file.wav"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())

            # Predict the emotion
            predicted_emotion = predict_emotion(temp_file)

            if predicted_emotion:
                st.audio(temp_file, format='audio/wav')
                st.success(f"Predicted Emotion: {predicted_emotion}")
            #else:
            #    st.error("Failed to predict emotion. Please try another file.")




