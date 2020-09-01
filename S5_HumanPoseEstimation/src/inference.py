import torch
from PIL import Image

from torchvision import transforms
from .pose_resnet import *
from operator import itemgetter
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
            [7, 6],
# LOWER BODY
            [6, 2],
            [2, 1],
            [1, 0],

            [6, 3],
            [3, 4],
            [4, 5],
# ARMS
            [7, 12],
            [12, 11],
            [11, 10],
            [7, 13],
            [13, 14],
            [14, 15]
]

class HPEInference():
    """ Docstring
    """

    def __init__(self,cfg):

        self.model = get_pose_net(cfg, is_train=False)
        self.model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')))
        self.IMAGE_SIZE = cfg.MODEL.IMAGE_SIZE

        self.OUT_WIDTH,self.OUT_HEIGHT = cfg.MODEL.EXTRA.HEATMAP_SIZE

        self.transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        self.output = None

    def gen_output(self,img):
        tr_img = self.transform(img)
        self.output = self.model(tr_img.unsqueeze(0))
        self.output = self.output.squeeze(0)
        _, OUT_HEIGHT, OUT_WIDTH = self.output.shape
        print(self.output.shape)

        return self.output

    def heat_map(self,img):
        plt.figure(figsize=(15, 15))

        if self.output is None:
            self.gen_output(img)

        for idx, pose_layer in enumerate(get_detached(self.output)):
            # print(pose_layer.shape)
            plt.subplot(4, 4, idx + 1)
            plt.title(f'{idx} - {JOINTS[idx]}')
            plt.imshow(img.resize((self.OUT_WIDTH, self.OUT_HEIGHT)), cmap='gray', interpolation='bicubic')
            plt.imshow(pose_layer, alpha=0.5, cmap='jet', interpolation='bicubic')
            plt.axis('off')
        plt.show()

    def vis_pose(self,img,threshold = 0.5):

        if self.output is None:
            self.gen_output(img)

        THRESHOLD = threshold
        OUT_SHAPE = (self.OUT_HEIGHT, self.OUT_WIDTH)
        image_p = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        pose_layers = get_detached(x=output)
        key_points = list(get_keypoints(pose_layers=pose_layers))
        is_joint_plotted = [False for i in range(len(JOINTS))]
        for pose_pair in POSE_PAIRS:
            from_j, to_j = pose_pair

            from_thr, (from_x_j, from_y_j) = key_points[from_j]
            to_thr, (to_x_j, to_y_j) = key_points[to_j]

            IMG_HEIGHT, IMG_WIDTH = self.IMAGE_SIZE

            from_x_j, to_x_j = from_x_j * IMG_WIDTH / OUT_SHAPE[0], to_x_j * IMG_WIDTH / OUT_SHAPE[0]
            from_y_j, to_y_j = from_y_j * IMG_HEIGHT / OUT_SHAPE[1], to_y_j * IMG_HEIGHT / OUT_SHAPE[1]

            from_x_j, to_x_j = int(from_x_j), int(to_x_j)
            from_y_j, to_y_j = int(from_y_j), int(to_y_j)

            if from_thr > THRESHOLD and not is_joint_plotted[from_j]:
                # this is a joint
                cv2.ellipse(image_p, (from_x_j, from_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
                is_joint_plotted[from_j] = True

            if to_thr > THRESHOLD and not is_joint_plotted[to_j]:
                # this is a joint
                cv2.ellipse(image_p, (to_x_j, to_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
                is_joint_plotted[to_j] = True

            if from_thr > THRESHOLD and to_thr > THRESHOLD:
                # this is a joint connection, plot a line
                cv2.line(image_p, (from_x_j, from_y_j), (to_x_j, to_y_j), (255, 74, 0), 3)