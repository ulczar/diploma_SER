import streamlit as st


st.set_page_config(page_title="Speech Emotion Recognition",
                   page_icon=":microphone:",
                   layout="wide")

html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile

import os
import io

import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep

import librosa
import speech_recognition as sr
import pyaudio

from emotion_recognizer import SpeechEmotionRecognizer
from speech_recognizer import SpeechToText

from transformers import AutoTokenizer, AutoModel
from transformers import pipeline



@st.cache_resource
def get_sent():
    classifier = pipeline("sentiment-analysis", model="Tatyana/rubert-base-cased-sentiment-new")
    return classifier

@st.cache_resource
def get_model_trans():
    model_trans = SpeechToText()
    return model_trans

@st.cache_resource
def get_model_ser():
    model_ser = SpeechEmotionRecognizer()
    return model_ser

with st.spinner('Загрузка моделей...'):
    model_trans = get_model_trans()
    ser = get_model_ser()
    classifier_sent = get_sent()

st.title("Speech Emotion Recognition App")
st.sidebar.header("Options")
options = ["Demo Audio Files", "Upload Audio", "Real-time speech analyzer"]
choice = st.sidebar.radio("Select an option:", options)
with st.sidebar:
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"), unsafe_allow_html=True)
if choice == "Demo Audio Files":
    demo_files = {"Angry": "demo/0c72e24442279391d00e8c454b429bbc.wav",
                  "Positive": "demo/24f1c91fa9f00fbbb9c1f0c5b8ab5aba.wav",
                  "Sad": "demo/25a5f834339f83fa7b7a563ae73c1162.wav",
                  "Neutral":"demo/83f2f0a09abacefa617c031a6c8e5cda.wav"}
    
    col1, col2= st.columns(2)
    with col1:
        emo_choice = st.radio("Select an emotion:", list(demo_files.keys()), horizontal=True)
        emo_file = demo_files[emo_choice]
        audio_file = open(emo_file, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)


    if st.button("Predict Emotion"):
        audio, _ = librosa.load(emo_file, sr=16000)
        prediction = ser.predict(audio)
        st.write("Predicted Emotion:", prediction)

elif choice == "Upload Audio":
    file = st.file_uploader("Upload an audio file:", type=["wav", "mp3", "ogg"])
    col1, col2= st.columns(2)
    if file is not None:
        with col1:
            audio_bytes = file.read()
            st.audio(audio_bytes)
            audio_data = np.frombuffer(file.getvalue(), dtype=np.int16)
            audio_data = audio_data.astype('float32')

        if st.button("Predict Emotion"):
            prediction = ser.predict(audio_data)
            st.write("Predicted Emotion:", prediction)

elif choice == "Real-time speech analyzer":
    st.write("Click the button to initiate recording from your computer's microphone")
    labels_colors_map = {
            "angry":"#d62828",
            "positive":"#fcbf49",
            "neutral":"#eae2b7",
            "sad":"#003049",
        None:"white"
       
        }
    def make_chart(time, emotions):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=emotions, mode='lines+markers', line_shape='spline',
                                 marker=dict(
                                    color=[labels_colors_map[em] for em in emotions],
                                    size=30,

                                   ), 
                                 line=dict(
                                     color='#cecebc'
                                 )))
        fig.update_layout(height=400, 
                          width = 1500,
                          xaxis_title='time', 
                          yaxis_title='emotion', 
                          xaxis_range=[time[0], time[-1]], 
                          yaxis_range=[-1, 4],
                              title={
                                'text': "Emotions through time",
                                'y':0.9,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'}
                         )
        
        st.write(fig)
    
    container = st.empty()
    button_A = container.button('Start Recording 🎙️')
    if button_A:
        container.empty()
        button_B = container.button('Stop Recording ⏹️')
        rec_spot = st.empty()
        text_spot = st.empty()
        container_pred = st.empty()
        container_sent = st.empty()
        emotions = []
        plot_spot = st.empty()
        phrase_time = None
        last_sample = bytes()
        data_queue = Queue()
        recorder = sr.Recognizer()
        recorder.energy_threshold = model_trans.energy_threshold
        recorder.dynamic_energy_threshold = False
        recorder.pause_threshold = 0.7
        source = sr.Microphone(sample_rate=16000)
        temp_file = NamedTemporaryFile().name
        transcription = ['']
        #with source:
        #    recorder.adjust_for_ambient_noise(source)
        def record_callback(_, audio: sr.AudioData) -> None:
            data = audio.get_raw_data()
            data_queue.put(data)

        recorder.listen_in_background(source, record_callback, phrase_time_limit=model_trans.record_timeout)
        sleep(3)
        rec_spot.write("🔴 Recording...")

        while not button_B:
            t = False
            timeout = 0
            while True and not button_B:
                try:
                    now = datetime.utcnow()
                    if not data_queue.empty():
                        phrase_complete = False
                        if phrase_time and now - phrase_time > timedelta(seconds=model_trans.phrase_timeout):
                            last_sample = bytes()
                            phrase_complete = True
                        phrase_time = now

                        while not data_queue.empty():
                            data = data_queue.get()
                            last_sample += data
  
                        audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                        b = audio_data.get_wav_data()
                        wav_data = io.BytesIO(b)

                        with open(temp_file, 'w+b') as f:
                            f.write(wav_data.read())
 
                        result = model_trans.audio_model.transcribe(temp_file, fp16=torch.cuda.is_available(), language='russian')
                        text = result['text'].strip()
        
                        if phrase_complete:
                            audio_data = np.frombuffer(b, dtype=np.int16)
                            audio_data = audio_data.astype('float32')
                            intervals = librosa.effects.split(audio_data, top_db=33)
                            speech_intervals = [interval for interval in intervals if (interval[1]-interval[0])/16000 >= 0.5]
                            if len(speech_intervals) == 0:
                                prediction = None
                            else:
                                prediction = model_trans.ser.predict(audio_data)

                            emotions.append(prediction)
                            container_pred.empty()
                            prediction = prediction if prediction else 'no voice detected'
                            container_pred.markdown(f"Prediction: **{prediction}**")
                            with plot_spot:
                                make_chart(list(range(len(emotions)))[-15:], emotions[-15:])
                                
                            transcription.append(text)
                        else:
                            transcription[-1] = text
                            
                        container_sent.write(f"Sentiment prediction: **{classifier_sent(' '.join(transcription[-2:]))[0]['label'].lower()}**")    
                        os.system('cls' if os.name == 'nt' else 'clear')
                        with text_spot:
                            text_spot.empty()
                            text_spot.write(' '.join(transcription[-7:]))
                        sleep(0.25)
                        t=False
                        timeout=0

                    elif not t:
                        if timeout>5:
                            emotions.append(None)
                            container_pred.empty()
                            prediction = 'no voice detected'
                            container_pred.markdown(f"Prediction: **{prediction}**")
                            with plot_spot:
                                make_chart(list(range(len(emotions)))[-15:], emotions[-15:])
                            t=True
                            timeout=0
                        else:
                            sleep(0.25)
                            timeout+=0.25
                                
                            
                except KeyboardInterrupt:
                    break

            print("\n\nTranscription:")
            for line in transcription:
                st.write(line)
            source.stop()
            container.empty()
            button_A = container.button('Start Recording 🎙️')