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
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from requests_toolbelt.multipart import decoder

# define env bariables if there are not existing
MODEL_PATH = 'models/VAE_jit2.pt'


if not os.path.isfile(MODEL_PATH):
    print("Loading Model")
    model = torch.jit.load(MODEL_PATH)
    print("Model Loaded...")

model = torch.jit.load(MODEL_PATH)

# Transforms for data loader
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5)) ])

dataset_test = datasets.ImageFolder('data/test/', transform=transform)
test_loader = DataLoader(dataset=dataset_test, 
                         batch_size=1, 
                         shuffle=True)

headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}

def variational_auto_encoder(event, context):
    """Generate Indian car image using api.

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

        model.eval()
        x, _ = test_loader.__iter__().next() 

        # x = x.to(DEVICE)
        # x = x[:16].to(DEVICE)
        out, kl_div = model(x)
        x = (x.data + 1) / 2

        img_out = (np.clip((np.transpose(out.detach().numpy(), [1,2,0])+1)/2.0,0,1)*255).astype(np.uint8)
        buffered = BytesIO()
        img_out.save(buffered, format="JPEG")
        
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
