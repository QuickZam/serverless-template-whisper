import torch
import whisper
import os
import base64
import urllib.request
from pytube import YouTube
from io import BytesIO
from stable_whisper import modify_model

def init():
    global model
    
    model = whisper.load_model("large-v1")
    modify_model(model)

def inference(model_inputs:dict) -> dict:
    global model

    link = model_inputs.get('link', None)

    if 'tinyurl' in link: 
        path = urllib.request.urlretrieve(link, f"{link.split('/')[-1]}.mp4")[0]

    else:
        yt = YouTube(link)
        path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()

    translate_options = dict(task="translate", suppress_silence=True, ts_num=16, lower_quantile=0.05, lower_threshold=0.1)
    result = model.transcribe(path, **translate_options)
    os.remove(path)

    return result 
    
