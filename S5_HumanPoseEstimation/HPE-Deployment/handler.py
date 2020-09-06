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
import copy
import cv2

from requests_toolbelt.multipart import decoder

import onnxruntime


print("Import End...")

get_detached = lambda x: copy.deepcopy(x.cpu().detach().numpy())
get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])

JOINTS = ['r-ankle', 'r-knee', 'r-hip', 'l-hip',
        'l-knee', 'l-ankle', 'pelvis', 'thorax',
        'upper-neck', 'head-top', 'r-wrist', 'r-elbow',
        'r-shoulder', 'l-shoulder', 'l-elbow', 'l-wrist']

POSE_PAIRS = [
# UPPER BODY
            [9, 8], [8, 7], [7, 3], [7, 2],
# LOWER BODY
            [6, 2], [2, 1], [1, 0], [6, 3], [3, 4], [4, 5],
# ARMS
            [8, 12], [12, 11], [11, 10], [8, 13], [13, 14], [14, 15]
]

POSE_PAIRS_COL = [
# UPPER BODY
    (65,190,115), (50,55,235), (110,230,255), (195,125,40), 
# LOWER BODY
    (180,75,160), (255,225,110), (65,190,115), (180,75,160), (50,55,240), (195,125,40),
# ARMS
    (180,75,160), (110,230,255), (195,125,40), (255,225,110), (50,55,240), (65,190,115) 
]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# define env bariables if there are not existing
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ \
    else 'tsai-assignment-models-s5'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ \
    else 'simple_pose_estimation.8bit_quantized.onnx'

print('Downloading model...')

# s3 = boto3.client('s3')
s3 = boto3.resource('s3')

try:
    if not os.path.isfile(MODEL_PATH):
        print("Loading ONNX Model file in temp directory")
        onnx_file = s3.Object(S3_BUCKET, 
                              MODEL_PATH).download_file('/tmp/simple_pose_estimation.8bit_quantized.onnx')
        print("Model Loaded...")
except Exception as e:
    print(repr(e))
    raise(e)


class HPEInference_onnx():
    """ Docstring
    """

    def __init__(self):

        self.ort_session = onnxruntime.InferenceSession('/tmp/simple_pose_estimation.8bit_quantized.onnx')
        self.OUT_WIDTH,self.OUT_HEIGHT = 64,64

    def transforms(self, img):
        sized = cv2.resize(img, (256,256))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        sized = sized[np.newaxis, ...]

        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])

        img_data = np.stack(sized).transpose(0, 3, 1, 2)

        #normalize
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img_data.shape).astype('float32')

        for i in range(img_data.shape[0]):
            for j in range(img_data.shape[1]):
                norm_img_data[i,j,:,:] = (img_data[i,j,:,:]/255 - mean_vec[j]) / stddev_vec[j]

        #add batch channel
        norm_img_data = norm_img_data.reshape(-1, 3, 256, 256).astype('float32')
        return norm_img_data

    def gen_output(self,img):
        tr_img = self.transforms(img)
        # compute ONNX Runtime output prediction
        ort_inputs = {self.ort_session.get_inputs()[0].name: tr_img}
        ort_outs = self.ort_session.run(None, ort_inputs)

        print(np.array(ort_outs).shape)
        ort_outs = np.array(ort_outs[0][0])

        return ort_outs

    def vis_pose(self,img,threshold = 0.5):
        since = time.time()

        output = self.gen_output(img)

        THRESHOLD = threshold
        OUT_SHAPE = (self.OUT_HEIGHT, self.OUT_WIDTH)
        #image_p = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        image_p = img

        IMG_HEIGHT, IMG_WIDTH, _ = image_p.shape

        scale_x = IMG_WIDTH / OUT_SHAPE[0]
        scale_y = IMG_HEIGHT / OUT_SHAPE[1]

        pose_layers = output
        key_points = list(get_keypoints(pose_layers=pose_layers))
        key_points = [(thres,(int(x*scale_x),int(y*scale_y))) for thres,(x,y) in key_points]

        i=0
        for from_j, to_j in POSE_PAIRS:
            from_thr, (from_x_j, from_y_j) = key_points[from_j]
            to_thr, (to_x_j, to_y_j) = key_points[to_j]

            if from_thr > THRESHOLD and to_thr > THRESHOLD:
                # this is a joint connection, plot a line
                cv2.line(image_p, (from_x_j, from_y_j), (to_x_j, to_y_j), POSE_PAIRS_COL[i], 3)
            i+=1

        for thres,(x,y) in key_points:
            if thres > THRESHOLD:
                # this is a joint
                cv2.ellipse(image_p, (x, y), (5, 5), 0, 0, 360, (255, 255, 255), cv2.FILLED)

        time_elapsed = time.time() - since
        print('Inference complete in {:4.2f}ms'.format(
        time_elapsed*1000))

        return cv2.imencode(".jpg", cv2.cvtColor(image_p, cv2.COLOR_BGR2RGB))

headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}

def human_pose_estimation(event, context):
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

        hpe_infer_onnx = HPEInference_onnx()
        img_out = hpe_infer_onnx.vis_pose(img, 0.4)

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