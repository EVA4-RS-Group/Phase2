#import torch
from PIL import Image

from torchvision import transforms
from operator import itemgetter
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import onnxruntime

get_detached = lambda x: copy.deepcopy(x.cpu().detach().numpy())
get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])

JOINTS = ['r-ankle', 'r-knee', 'r-hip', 'l-hip',
        'l-knee', 'l-ankle', 'pelvis', 'thorax',
        'upper-neck', 'head-top', 'r-wrist', 'r-elbow',
        'r-shoulder', 'l-shoulder', 'l-elbow', 'l-wrist']

POSE_PAIRS = [
# UPPER BODY
            [9, 8],
            [8, 7],
            [7, 3],
            [7, 2],
# LOWER BODY
            [6, 2],
            [2, 1],
            [1, 0],

            [6, 3],
            [3, 4],
            [4, 5],
# ARMS
            [8, 12],
            [12, 11],
            [11, 10],
            [8, 13],
            [13, 14],
            [14, 15]
]

POSE_PAIRS_COL = [
    (113,191,66),
    (237,56,51),
    (114,192,67),
    (157,74,182),
    (253,226,107),
    (41,125,194),
    (41,126,195),
    (157,74,182),
    (113,191,66),
    (237,56,51),
    (114,192,67),
    (157,74,182),
    (253,226,107),
    (41,125,194),
    (157,74,182),
    (113,191,66)
]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class HPEInference_onnx():
    """ Docstring
    """

    def __init__(self,model_name):

        self.ort_session = onnxruntime.InferenceSession(model_name)
        self.IMAGE_SIZE = [256,256]

        self.OUT_WIDTH,self.OUT_HEIGHT = 64,64

        self.transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

    def gen_output(self,img):
        tr_img = self.transform(img)
        # compute ONNX Runtime output prediction
        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(tr_img.unsqueeze(0))}
        ort_outs = self.ort_session.run(None, ort_inputs)

        print(np.array(ort_outs).shape)
        ort_outs = np.array(ort_outs[0][0])

        return ort_outs

    def vis_pose(self,img,threshold = 0.5):
        since = time.time()

        output = self.gen_output(img)

        THRESHOLD = threshold
        OUT_SHAPE = (self.OUT_HEIGHT, self.OUT_WIDTH)
        image_p = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

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
                cv2.ellipse(image_p, (x, y), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)



        time_elapsed = time.time() - since
        print('Inference complete in {:4.2f}ms'.format(
        time_elapsed*1000))

        return Image.fromarray(cv2.cvtColor(image_p, cv2.COLOR_RGB2BGR))



