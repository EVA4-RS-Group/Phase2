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

import torch
from requests_toolbelt.multipart import decoder
# import onnxruntime

print("Import End...")

# define env bariables if there are not existing
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ \
    else 'roshantac-bucket1'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ \
    else 'VAE_jit.pt'


s3 = boto3.client('s3')

try:
    if not os.path.isfile(MODEL_PATH):
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print("Loading Model")
        model = torch.jit.load(bytestream)
        print("Model Loaded...")
except Exception as e:
    print(repr(e))
    raise(e)


# def transform_image(image_bytes):
#     """Transform the image for pre-trained model.

#     Transform the image which includes resizing, centercrop and normalize.

#     Args:
#         image_bytes: Input image in bytes

#     Returns:
#         Tensor

#     Raises:
#         Except: An error occurred accessing the bytes.
#     """
#     try:
#         transformations = transforms.Compose([
#             transforms.Resize(255),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#         image = Image.open(io.BytesIO(image_bytes))
#         return transformations(image).unsqueeze(0)
#     except Exception as e:
#         print(repr(e))
#         raise(e)


# def imagenet1000_classidx_to_label(class_idx: int) -> str:
#     """Convert class index to class labels, Imagenet.

#     Converting predicted class index to class labels, Imagenet.

#     Args:
#         class_idx: int, predicted class index.

#     Returns:
#         A dict mapping keys to the corresponding table row data
#         fetched. Each row is represented as a tuple of strings. For
#         example:

#         {b'Serak': ('Rigel VII', 'Preparer'),
#          b'Zim': ('Irk', 'Invader'),
#          b'Lrrr': ('Omicron Persei 8', 'Emperor')}

#         Returned keys are always bytes.  If a key from the keys argument is
#         missing from the dictionary, then that row was not found in the
#         table (and require_all_keys must have been False).

#     Raises:
#         Exception: An error occurred accessing the json file.
#     """
#     try:
#         with open("data/imagenet1000_clsidx_to_labels.json", "r") as f:
#             map = json.loads(f.read())
#             return map[str(class_idx)]
#     except Exception as e:
#         print(repr(e))
#         return "Class Not Found"


def get_prediction(image_bytes):
    """Prediction from pre-trained model.

    Inferecing using pre-trained model

    Args:
        image_bytes: Transformed image_bytes

    Returns:
        int, predicted class index from imagenet
    """
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()


def variational_auto_encodder(event, context):
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
        # # print(event['body'])
        # body = base64.b64decode(event["body"])
        print('BODY LOADED')
        
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = get_prediction(image_bytes=picture.content)
        print(prediction)

        prediction_label = imagenet1000_classidx_to_label(prediction)

        filename = (picture
                    .headers[b'Content-Disposition']
                    .decode().split(';')[1].split('=')[1])

        if len(filename) < 4:
            filename = (picture
                        .headers[b'Content-Disposition']
                        .decode().split(';')[2].split('=')[1])

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
                },
            "body": json.dumps({'file': filename.replace('"', ''),
                                'predicted': f"{prediction}, {prediction_label}"})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            "body": json.dumps({"error": repr(e)})
        }


# model_name='/tmp/'+ MODEL_PATH

# print('Downloading model...')

# # s3 = boto3.client('s3')
# s3 = boto3.resource('s3')

# try:
#     if not os.path.isfile(MODEL_PATH):
#         print("Loading ONNX Model file in temp directory")
#         onnx_file = s3.Object(S3_BUCKET, 
#                               MODEL_PATH).download_file(model_name)
#         print("Model Loaded...")
# except Exception as e:
#     print(repr(e))
#     raise(e)

# headers = {
#     "Content-Type": "application/json",
#     "Access-Control-Allow-Origin": "*",
#     "Access-Control-Allow-Credentials": True,
# }

# def get_sample_image(n_noise=100, n_samples=64):
#     """
#         save sample 100 images
#     """
#     n_rows = 5

#     ort_session = onnxruntime.InferenceSession(model_name)
#     rand_inp = np.array(np.random.rand(n_samples,n_noise)*2-1,dtype='f')
#     ort_inputs = {ort_session.get_inputs()[0].name: rand_inp}
#     ort_outs = ort_session.run(None,ort_inputs)

#     x_fake = np.concatenate([np.concatenate([ort_outs[0][n_rows*j+i] for i in range(n_rows)], axis=1) for j in range(n_rows)], axis=2)
#     #img = Image.fromarray((np.clip((np.transpose(x_fake, [1,2,0])+1)/2.0,0,1)*255).astype(np.uint8))
#     img = (np.clip((np.transpose(x_fake, [1,2,0])+1)/2.0,0,1)*255).astype(np.uint8)

#     return img


# def generative_adversarial_network(event, context):
#     """Classify image using api.
#     Function is called from this template python: handler.py
#     Args:
#         event: Dictionary containing API inputs 
#         context: Dictionary
#     Returns:
#         dictionary: API response
#     Raises:
#         Exception: Returning API repsonse 500
#     """
#     try:
#         # content_type_header = event['headers']['content-type']
#         # print(event['body'])
#         # body = base64.b64decode(event["body"])
#         print('BODY LOADED')

#         img_out = get_sample_image(n_noise=256, n_samples=25)
#         #print(img_out.shape)
#         #print(img_out.dtype)
#         #pil_img = Image.fromarray((img_out * 255).astype(np.uint8))
#         #img_out = np.array(img_out, dtype=img_out.dtype, order='C')
#         #buffered = io.BytesIO(img_out)

#         print('INFERENCING SUCCESSFUL, RETURNING IMAGE')
#         fields = {"file0": ("file0", base64.b64encode(img_out.tobytes()).decode("utf-8"), "image/jpg",)}

#         return {"statusCode": 200, "headers": headers, "body": json.dumps(fields)}

#     except ValueError as ve:
#         # logger.exception(ve)
#         print(ve)
#         return {
#             "statusCode": 422,
#             "headers": headers,
#             "body": json.dumps({"error": repr(ve)}),
#         }
#     except Exception as e:
#         # logger.exception(e)
#         print(e)
#         return {
#             "statusCode": 500,
#             "headers": headers,
#             "body": json.dumps({"error": repr(e)}),
#         }
