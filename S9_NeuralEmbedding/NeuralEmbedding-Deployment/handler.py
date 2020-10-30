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
import copy
import numpy as np
import pickle
import torch
import torchtext
from torchtext import data

#from requests_toolbelt.multipart import decoder

print("Importing Packages Done...")


# define env bariables if there are not existing
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ \
    else 'tsai-assignment-models-s9'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ \
    else 'tut3-model_cpu.pt'
TEXT_VOCAB = os.environ['TEXT_VOCAB'] if 'TEXT_VOCAB' in os.environ \
    else 'TEXT_vocab_stoi.pickle'

print('Downloading model...')

s3 = boto3.client('s3')

try:
    if not os.path.isfile(MODEL_PATH):
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        traced_model = torch.jit.load(bytestream)
        print("Model Loaded...")
    if not os.path.isfile(TEXT_VOCAB):
        obj = s3.get_object(Bucket=S3_BUCKET, Key=TEXT_VOCAB)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading TEXT VOCAB")
        TEXT_vocab_stoi = pickle.load(bytestream)
        print(f"TEXT VOCAB Loaded...{len(TEXT_vocab_stoi)}")

    # if not os.path.isfile(TEXT_VOCAB):
    #     with io.BytesIO() as data:
    #         s3r.Bucket(S3_BUCKET).download_fileobj(TEXT_VOCAB, data)
    #         data.seek(0)    # move back to the beginning after writing
    #         text_vocal_stoi = pickle.load(data)

except Exception as e:
    print(repr(e))
    raise(e)

def generate_bigrams(x):
    """Docstring
    """
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

def predict_sentiment_jit(model, sentence):
    """Docstring
    """
    model.eval()
    tokenized = generate_bigrams([tok for tok in sentence.split()])
    indexed = [TEXT_vocab_stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1)
    #print(f"tensor size :{tensor.shape}")
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}

def neural_embedding(event, context):
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
        content_type_header = event['headers']['content-type']
        # print(event['body'])
        body = base64.b64decode(event["body"])
        print('BODY LOADED')

        # picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        # img = cv2.imdecode(np.frombuffer(picture.content, np.uint8), -1)

        # hpe_infer_onnx = HPEInference_onnx()
        # err, img_out = hpe_infer_onnx.vis_pose(img, 0.4)
        # print('INFERENCING SUCCESSFUL, RETURNING IMAGE')

        prediction_label = "Negative"
        input_text = "This film is terrible"

        prediction = predict_sentiment_jit(traced_model, input_text)
        prediction_label = "Positive" if prediction > 0.5 else "Negative"

        fields = {'input': input_text,
                  'predicted': f"Predicted sentiment : {prediction_label}"}

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