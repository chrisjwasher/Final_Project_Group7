import os
from itertools import groupby
from operator import concat
from pickle import FALSE

import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import json
import seaborn as sns
import plotly.graph_objects as go
import joblib
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
import seaborn as sns

from IPython.lib.display import Audio
from pygments.lexer import bygroups


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

    page = st.sidebar.radio('', ["📊EDA", "📈Model"])


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



# =========================================================================================================
#                                   EDA
# =========================================================================================================
# st.sidebar.title('Choose Sector')
# page=st.sidebar.radio(['','Data Visualization', 'Prediction'])

# ---------------------------------------EDA--------------------------------------------------------

if page == '📊EDA':
    st.markdown('Data')
    st.dataframe(data,height=300)


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



# ------------------------------------------Prediction------------------------------------------

elif page == '📈Model':

    # %%
    import pandas as pd
    import warnings
    import joblib

    #warnings.filterwarnings("ignore")


