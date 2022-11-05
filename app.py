import torch
import whisper
import os
import base64
from pytube import YouTube
from io import BytesIO

def init():
    global model
    
    model = whisper.load_model("large")

def inference(model_inputs:dict) -> dict:
    global model

    link = model_inputs.get('link', None)
    yt = YouTube(link)
    path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    result = model.transcribe(path)
    output = {"text":result["text"]}
    os.remove(path)

    return output
