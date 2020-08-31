import torch
from PIL import Image

from torchvision import transforms
from .pose_resnet import get_pose_net


class HPEInference():
    def __init__(self,cfg):

        self.model = get_pose_net(cfg, is_train=False)
        self.model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')))
        self.image_size = cfg.MODEL.IMAGE_SIZE

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        self.output = None

    def gen_output(self,img):
        tr_img = transform(img)
        output = self.model(tr_img.unsqueeze(0))
        output = output.squeeze(0)
        _, OUT_HEIGHT, OUT_WIDTH = output.shape
        output.shape
        print(output.shape)

        return output
