import torch
from PIL import Image

from torchvision import transforms
from .pose_resnet import *
from operator import itemgetter
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

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
# UPPER BODY
    (65,190,115), # Green
    (50,55,235), # Red
    (110,230,255), # Yellow
    (195,125,40), # Blue
# LOWER BODY
    (180,75,160), # Purple
    (255,225,110), # Cyan
    (65,190,115), # Green

    (180,75,160), # Purple
    (50,55,240), # Red
    (195,125,40), # Blue
# ARMS
    (180,75,160), # Purple
    (110,230,255), # Yellow
    (195,125,40), # Blue
    (255,225,110), # Cyan
    (50,55,240), # Red
    (65,190,115) # Green
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

    def gen_output(self,img):
        tr_img = self.transform(img)
        output = self.model(tr_img.unsqueeze(0))
        output = output.squeeze(0)
        _, OUT_HEIGHT, OUT_WIDTH = output.shape
        print(output.shape)

        return output

    def heat_map(self,img):
        since = time.time()
        plt.figure(figsize=(15, 15))

        output = self.gen_output(img)

        for idx, pose_layer in enumerate(get_detached(output)):
            # print(pose_layer.shape)
            plt.subplot(4, 4, idx + 1)
            plt.title(f'{idx} - {JOINTS[idx]}')
            plt.imshow(img.resize((self.OUT_WIDTH, self.OUT_HEIGHT)), cmap='gray', interpolation='bicubic')
            plt.imshow(pose_layer, alpha=0.5, cmap='jet', interpolation='bicubic')
            plt.axis('off')
        plt.show()

        time_elapsed = time.time() - since
        print('Inference complete in {:4.2f}ms'.format(
        time_elapsed*1000))

    def vis_pose(self,img,threshold = 0.5):
        since = time.time()

        output = self.gen_output(img)

        THRESHOLD = threshold

        image_p = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        OUT_SHAPE = (self.OUT_HEIGHT, self.OUT_WIDTH)
        IMG_HEIGHT, IMG_WIDTH, _ = image_p.shape

        scale_x = IMG_WIDTH / OUT_SHAPE[0]
        scale_y = IMG_HEIGHT / OUT_SHAPE[1]

        pose_layers = get_detached(x=output)
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

        return Image.fromarray(cv2.cvtColor(image_p, cv2.COLOR_RGB2BGR))

    def export_onnx_model(self, model_name = "simple_pose_estimation.onnx",quantization = False):
        torch_model = copy.deepcopy(self.model)
        batch_size = 1
        rand_inp = torch.randn(batch_size, 3, *self.IMAGE_SIZE, requires_grad=True)

        # Export the model
        torch.onnx.export(torch_model,               # model being run
                        rand_inp,                    # model input (or a tuple for multiple inputs)
                        model_name,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=10,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                      'output' : {0 : 'batch_size'}})
        
        if quantization:
            import onnx
            from onnxruntime.quantization import quantize
            
            onnx_model = onnx.load(model_name)
            quantized_model = quantize(onnx_model)
            onnx.save(quantized_model, model_name.replace(".onnx",".8bit_quantized.onnx"))