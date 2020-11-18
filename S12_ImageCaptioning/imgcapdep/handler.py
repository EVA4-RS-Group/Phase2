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
from imageio import imread
import torch
from caption import *

#from requests_toolbelt.multipart import decoder

print("Importing Packages Done...")


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
    





headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}


model = 'img_cap_flicker8.pth.tar'
word_map_file = 'WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'
beam_size = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model
checkpoint = torch.load(model, map_location=str(device))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()


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
        img = cv2.imdecode(np.frombuffer(picture.content, np.uint8), -1)
        output = genCap(img)

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
