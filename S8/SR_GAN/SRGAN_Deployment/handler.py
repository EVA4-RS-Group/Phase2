"""Code to download pre-trained pytorch model.

Download the pre-trained model and convert into trace model
"""
try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import base64
import numpy as np
# from PIL import Image
import cv2

from requests_toolbelt.multipart import decoder
import onnxruntime

print("Import End...")

model_name='models/SRGAN.onnx'

print('Downloading model...')



headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}

def getSRImage(inptImg):
    
    ort_session = onnxruntime.InferenceSession(model_name) 
    rand_inp = transforms(inptImg)
    ort_inputs = {ort_session.get_inputs()[0].name: rand_inp}
    ort_outs = ort_session.run(None,ort_inputs)
    return np.transpose(ort_outs[0][0,:,:,:], [1,2,0])


def transforms(img):
    sized = cv2.resize(img, (224,224))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)/255
    sized = sized[np.newaxis, ...]
    img_data = np.stack(sized).transpose(0, 3, 1, 2)
    return img_data.astype('float32')

def super_resolution_GAN(event, context):

    try:
        content_type_header = event['headers']['content-type']
        # print(event['body'])
        body = base64.b64decode(event["body"])
        print('BODY LOADED')
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        img = cv2.imdecode(np.frombuffer(picture.content, np.uint8), -1)

        out = getSRImage(img)
        #print(img_out.shape)
        #print(img_out.dtype)
        #pil_img = Image.fromarray((img_out * 255).astype(np.uint8))
        #img_out = np.array(img_out, dtype=img_out.dtype, order='C')
        #buffered = io.BytesIO(img_out)
        image_p = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        err, img_out = cv2.imencode(".jpg", image_p)

        print('INFERENCING SUCCESSFUL, RETURNING IMAGE')
        fields = {"file0": ("file0", base64.b64encode(img_out).decode("utf-8"), "image/jpg",)}

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
