import torch
from PIL import Image

from torchvision import transforms
from .pose_resnet import *
from operator import itemgetter
import copy
import matplotlib.pyplot as plt

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