import torch
import whisper
import os
import base64
import urllib.request
from pytube import YouTube
from io import BytesIO

def init():
    global model
    
    model = whisper.load_model("large")

def inference(model_inputs:dict) -> dict:
    global model

    link = model_inputs.get('link', None)

    if 'amazonaws' in link: 
        path = urllib.request.urlretrieve(link, f"{link.split('/')[-1]}.mp4")[0]

    else:
        yt = YouTube(link)
        path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()

    translate_options = dict(task="translate")
    result = model.transcribe(path, **translate_options)
    os.remove(path)

    return result 
