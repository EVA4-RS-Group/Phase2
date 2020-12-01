"""Doc string placeholder
"""
try:
    import unzip_requirements
except ImportError:
    pass

import boto3
import os
import io
import json
import base64
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import random
import glob

sample_file_list = list(glob.iglob('/sample_test_data/*.wav', recursive=True))

class SpeechRNN(torch.nn.Module):
  
  def __init__(self):
    super(SpeechRNN, self).__init__()
    
    self.lstm = torch.nn.GRU(input_size = 12, 
                              hidden_size= 256, 
                              num_layers = 2, 
                              batch_first=True)
    
    self.out_layer = torch.nn.Linear(256, 30)
    self.softmax = torch.nn.LogSoftmax(dim=1)
    
  def forward(self, x):
    out, _ = self.lstm(x)
    x = self.out_layer(out[:,-1,:])
    return self.softmax(x)

classes = ['right', 'left', 'go', 'on', 'one', 'zero', 'marvin', 'five', 
           'happy', 'bed', 'two', 'nine', 'yes', 'six', 'eight', 'sheila', 
           'wow', 'up', 'down', 'seven', 'dog', 'stop', 'tree', 'cat', 'four',
           'house', 'three', 'no', 'bird', 'off']



headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}

def speech_recog(event, context):
    """Classify image using api.

    Function is called from this template python: handler.py

    Args:
        event: Dictionary containing API inputs 
        context: Dictionary

    Returns:
        dictionary: API response

    Raises:
        Exception: Returning API repsonse 500
    """
    
    
    try:

        DEVICE=torch.device('cpu')
        model = SpeechRNN()
        model = model.to(DEVICE)
        model.load_state_dict(torch.load('weights_cpu_voicerec.pt', map_location=DEVICE))

        wav_file = random.choice(sample_file_list)
        waveform,_ = torchaudio.load(wav_file, normalization=True)
            
        # if the waveform is too short (less than 1 second) we pad it with zeroes
        if waveform.shape[1] < 16000:
            waveform = F.pad(input=waveform, pad=(0, 16000 - waveform.shape[1]), mode='constant', value=0)
                            
        mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=12, log_mels=True)
        mfcc = mfcc_transform(waveform).squeeze(0).transpose(0,1)
        x = mfcc.unsqueeze(0)

        model.eval()
        y = model(x)
        predicted_label = classes[y.max(1)[1].numpy().item()]

        input_text = wav_file.split("/")[-1]
        output = f'Prediction of input file {wav_file.split("/")[-1]} is {predicted_label}.'

        fields = {'input': input_text,
                  'predicted': output}

        return {"statusCode": 200, "headers": headers, "body": json.dumps(fields)}

    except ValueError as ve:
        # logger.exception(ve)
        print(ve)
        return {
            "statusCode": 422,
            "headers": headers,
            "body": json.dumps({"error": repr(ve)}),
        }
    except Exception as e:
        # logger.exception(e)
        print(e)
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": repr(e)}),
        }
