"""Code to download pre-trained pytorch model.

Download the pre-trained model and convert into trace model
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
from PIL import Image

from requests_toolbelt.multipart import decoder
import onnxruntime

print("Import End...")

# define env bariables if there are not existing
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ \
    else 'tsai-assignment-models-s6'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ \
    else 'GAN_Indian_cars.onnx'

model_name='/tmp/'+ MODEL_PATH

print('Downloading model...')

# s3 = boto3.client('s3')
s3 = boto3.resource('s3')

try:
    if not os.path.isfile(MODEL_PATH):
        print("Loading ONNX Model file in temp directory")
        onnx_file = s3.Object(S3_BUCKET, 
                              MODEL_PATH).download_file(model_name)
        print("Model Loaded...")
except Exception as e:
    print(repr(e))
    raise(e)

headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}

def get_sample_image(n_noise=100, n_samples=64):
    """
        save sample 100 images
    """
    n_rows = 5

    ort_session = onnxruntime.InferenceSession(model_name)
    rand_inp = np.array(np.random.rand(n_samples,n_noise)*2-1,dtype='f')
    ort_inputs = {ort_session.get_inputs()[0].name: rand_inp}
    ort_outs = ort_session.run(None,ort_inputs)

    x_fake = np.concatenate([np.concatenate([ort_outs[0][n_rows*j+i] for i in range(n_rows)], axis=1) for j in range(n_rows)], axis=2)
    #img = Image.fromarray((np.clip((np.transpose(x_fake, [1,2,0])+1)/2.0,0,1)*255).astype(np.uint8))
    img = (np.clip((np.transpose(x_fake, [1,2,0])+1)/2.0,0,1)*255).astype(np.uint8)

    return img


def generative_adversarial_network(event, context):
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
        # content_type_header = event['headers']['content-type']
        # print(event['body'])
        # body = base64.b64decode(event["body"])
        print('BODY LOADED')

        img_out = get_sample_image(n_noise=256, n_samples=25)
        #print(img_out.shape)
        #print(img_out.dtype)
        #pil_img = Image.fromarray((img_out * 255).astype(np.uint8))
        #img_out = np.array(img_out, dtype=img_out.dtype, order='C')
        #buffered = io.BytesIO(img_out)

        print('INFERENCING SUCCESSFUL, RETURNING IMAGE')
        fields = {"file0": ("file0", base64.b64encode(img_out.tobytes()).decode("utf-8"), "image/jpg",)}

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