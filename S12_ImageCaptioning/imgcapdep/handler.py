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
from caption import *
from PIL import Image
#from requests_toolbelt.multipart import decoder

print("Importing Packages Done...")


word_map_file = 'WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'
beam_size = 5

# define env bariables if there are not existing
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ \
    else 'roshan-eva12'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ \
    else 'img_cap_flicker8.pth.tar'

print('Downloading model...')

s3 = boto3.client('s3')

try:
    if not os.path.isfile(MODEL_PATH):
        DEVICE=torch.device('cpu')
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
 # Load model###################################################
        checkpoint = torch.load(bytestream, map_location=DEVICE)
        decoder = checkpoint['decoder']
        decoder = decoder.to(DEVICE)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(DEVICE)
        encoder.eval()

        print("Model Loaded...")
    

except Exception as e:
    print(repr(e))
    raise(e)
    





headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}







def genCap(img):
    # Load word map (word2ix)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, img, word_map, beam_size)
    alphas = torch.FloatTensor(alphas)
    words = [rev_word_map[ind] for ind in seq]
    words = words[1:len(words)-1]
    ret = ' '.join(words)
    return ret
    
    
def imgcap(event, context):
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
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        img = Image.open(io.BytesIO(picture.content))
        # img = cv2.imdecode(np.frombuffer(picture.content, np.uint8), -1)
        output = genCap(img)
        filename = (picture
                    .headers[b'Content-Disposition']
                    .decode().split(';')[1].split('=')[1])

        fields = {'file': filename.replace('"', ''),
                  'predicted': f"Caption : {output}"}

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
