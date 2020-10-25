try:
    import unzip_requirements  # noqa
except ImportError:
    pass

import base64
# import cv2
import json
import base64
import os
from requests_toolbelt.multipart import decoder, encoder
# import numpy as np
# from src.libs import utils
# from src.libs.logger import logger
# from src.models.facerec.facerec import FaceRecognition


headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}
# S3_BUCKET = "eva4-p2"

def style_transfer(event, context):
    try:

        # Loading the two images
        pic_details = []
        content_type_key = "Content-Type"
        if event["headers"].get(content_type_key.lower()):
            content_type_key = content_type_key.lower()
        content_type_header = event["headers"][content_type_key]

        body = base64.b64decode(event["body"])
        print('BODY LOADED')
        if type(event["body"]) is str:
            event["body"] = bytes(event["body"], "utf-8")

        pictures = decoder.MultipartDecoder(body, content_type_header)
        for part in pictures.parts:
            filename = get_picture_filename(part).replace('"', "")
            pic_details.append((part, filename))

        files = pic_details[0:max_files]

        if len(files) == 2:

            style_img = Image.open(io.BytesIO(files[0][0].content))
            content_img = Image.open(io.BytesIO(files[1][0].content))

            #im = Image.fromarray(out)
            buf = io.BytesIO()
            content_img.save(buf, format='JPEG')
            #byte_im = buf.getvalue()
            # buffered = io.BytesIO()
            # img_out.save(buffered, format="JPEG")
            # image_p = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            # err, img_out = cv2.imencode(".jpg", image_p)
            
            print('INFERENCING SUCCESSFUL, RETURNING IMAGE')
            fields = {"file0": ("file0", base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpg",)}
            #fields = {"file0": ("file0", base64.b64encode(swapped_img).decode("utf-8"), "image/jpg",)}

            return {"statusCode": 200, "headers": headers, "body": json.dumps(fields)}
        else:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": "Please pass exactly 2 files as input",
            }

    except ValueError as ve:
        logger.exception(ve)
        return {
            "statusCode": 422,
            "headers": headers,
            "body": json.dumps({"error": repr(ve)}),
        }
    except Exception as e:
        logger.exception(e)
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": repr(e)}),
        }
