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
import pickle
import torch
import torchtext
from torchtext import data
from model import *

#from requests_toolbelt.multipart import decoder

print("Importing Packages Done...")


# define env bariables if there are not existing
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ \
    else 'roshanevabucket'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ \
    else 'de_eng_translation_model'

print('Downloading model...')

s3 = boto3.client('s3')

try:
    if not os.path.isfile(MODEL_PATH):
    	DEVICE=torch.device('cpu')
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        model = make_model(len(SRC_vocab), len(TRG_vocab),
                   emb_size=256, hidden_size=256,
                   num_layers=1, dropout=0.2)

	model = model.to(DEVICE)
	model.load_state_dict(torch.load(bytestream, map_location=DEVICE))
        print("Model Loaded...")
    

except Exception as e:
    print(repr(e))
    raise(e)
    
with open('SRC_vocab.pickle', 'rb') as handle:
    SRC_vocab = pickle.load(handle)

with open('TRG_vocab.pickle', 'rb') as handle:
    TRG_vocab = pickle.load(handle)

SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
def translate(src): 
  tokenized = tokenize(src)#[tok.text for tok in spacy_de.tokenizer(src)]
  tokenized.append("</s>")
  # print(tokenized)
  indexed = [[SRC_vocab.stoi[t] for t in tokenized]]
  srcpr = [lookup_words(x, SRC_vocab) for x in indexed]
  # print("German : ",[" ".join(y) for y in srcpr][0])
  srcs = torch.LongTensor(indexed).to(DEVICE)
  length = torch.LongTensor([len(indexed[0])]).to(DEVICE)
  mask = (srcs != 0).unsqueeze(-2).to(DEVICE)
  # print(srcs)
  # print(mask)
  # print(length)
  pred, attention = greedy_decode(
    model, srcs, mask, length, max_len=25,
    sos_index=TRG_vocab.stoi[SOS_TOKEN],
    eos_index=TRG_vocab.stoi[EOS_TOKEN])
  # print(pred)
  english = lookup_words(pred, TRG_vocab)
  return  " ".join(english)



headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}

def de2en(event, context):
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

        input_text = "mein vater hörte sich auf seinem kleinen"
        input_text = json.loads(event['body'])["text"]
        print(json.loads(event['body']),input_text)

        output = translate(input_text)

        fields = {'input': input_text,
                  'predicted': f"English Translation : {output}"}

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
