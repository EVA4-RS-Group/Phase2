"""Code to define the dataset, dataloader and show sample images.

Author: Roshan
"""

import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets, models, transforms
import os
import glob
import csv
import random
import matplotlib.pyplot as plt
from pathlib import Path


## Section for face recognition
data_transforms_face_recog = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class LoadFaceDataset():
    """
    A class to dataset with its loader.

    ...

    Attributes
    ----------
    dataloaders : torch dataloader
    dataset_sizes : length of train and validation dataset
    class_names : class names

    Methods
    -------
    show_batch:
        Shows five sample images for verification of dataloaders
    """

    def __init__(self, data_dir, batch_size):
        """Initialize the class object.

        Dataset class ojbect
        """
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                               data_transforms_face_recog[x]) for x in ['train', 'val']}

        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x],
                                                           batch_size=self.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train','val']}

        self.class_names = self.image_datasets['train'].classes
        print(self.class_names)

    def show_batch(self, save_as="sample.jpg"):
        """Show five sample images for verification of dataloaders.

        Get item internal fuction
        """
        # Get a batch of training data
        inputs, classes = next(iter(self.dataloaders['train']))
        images = denormalize(inputs,mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)).cpu().numpy()

        counter=0
        fig = plt.figure(figsize=(15, 8))

        while(counter<7):
            ax = fig.add_subplot(1, 7, counter+1, xticks=[], yticks=[])
            img = images[counter]
            npimg = np.clip(np.transpose(img,(1,2,0)), 0, 1)
            ax.imshow(npimg, cmap='gray')
            ax.set_title(f'{self.class_names[classes[counter]]}', color= "blue",fontsize=16)
            counter+=1
        fig.tight_layout()  
        plt.show()
        fig.savefig(save_as)
        #imshow_save(out, save_as="sample.jpg",title=[class_names[int(x)] for x in classes[0:4]])

## Section for drone dataset

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.53713346, 0.58979464, 0.62127595],
                             [0.27420551, 0.25534403, 0.29759673])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.53713346, 0.58979464, 0.62127595],
                             [0.27420551, 0.25534403, 0.29759673])
    ]),
}



def split_test_train_data(rootImageDir, tstRatio = 0.1):
    """Split the classification folder into train and validation subset.

    Random split with given ratio for train and validation

    Args:
        rootImageDir: Input directory containing all image
        tstRatio: Percentage of train and validation subset

    Returns:
        None

    Raises:
        No Exception
    """
    tstFile = open('testData.csv','w', newline='')
    writertst =csv.writer(tstFile)
    trnFile = open('trainData.csv','w', newline='')
    writertrn =csv.writer(trnFile)
    dirs = os.listdir(rootImageDir)
    # Sorting the list as os.listdir return different order everytime.
    dirs.sort()
    print(dirs)
    dircnt = 0
    for fldr in dirs:
        files = glob.glob(rootImageDir+'/'+fldr+'/*.*')
        folderlen= len(files)
        print(folderlen)
        i = 0
        test=[]
        train=[]
        random.shuffle(files)
        for f in files:
            if (i < folderlen*(1 - tstRatio)):
                train.append(f)
            else:
                test.append(f)
            i+=1
        for filename in train:
            filename = filename.replace('\\','/')
            writertrn.writerow([filename,fldr,dircnt])
        for filename in test:
            filename = filename.replace('\\','/')
            writertst.writerow([filename,fldr,dircnt])
        dircnt +=1

class DroneDataset(Dataset):
    """Define Drone Dataset class to get images and labels .

    Dataset classes
    """

    def __init__(self, train=True, transform = None):
        """Initialize the class object.

        Dataset class ojbect
        """
        self.transform = transform
        if (train == True):
            data_file = open('trainData.csv','r')
        else:
            data_file = open('testData.csv','r')
        
        self.data = list(csv.reader(data_file))
        self.classes = ['Flying Birds', 'Large QuadCopters', 'Small QuadCopters', 'Winged Drones']

    def __len__(self):
        """Return length of dataset.

        Length internal fuction
        """
        return len(self.data)

    def __getitem__(self,idx):
        """Return dataset image and its label.

        Get item internal fuction
        """
        imgLoc, target =self.data[idx][0], int(self.data[idx][2])
        image = np.array(Image.open(imgLoc))
        # print(f"idx:{idx}")
        # print(f"target :{imgLoc}")
        # print(f"imgLoc :{self.classes[target]}")

        if (len(image.shape) == 2) or (len(image.shape)==3 and image.shape[-1]==1):
            image =np.stack((image,)*3, axis =-1)
        if self.transform :
            image = self.transform(Image.fromarray(image))
        return image, target

def denormalize(tensor, mean, std):
    """Denormalize the image for given mean and standard deviation.

    Args:
        tensor: Image tensor
        mean: Dataset mean
        std: Dataset standard deviation

    Returns:
        tensor

    Raises:
        No Exception
    """
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)

class LoadDataset():
    """
    A class to dataset with its loader.

    ...

    Attributes
    ----------
    dataloaders : torch dataloader
    dataset_sizes : length of train and validation dataset
    class_names : class names

    Methods
    -------
    show_batch:
        Shows five sample images for verification of dataloaders
    """

    def __init__(self, dir, tstRatio, batch_size):
        """Initialize the class object.

        Dataset class ojbect
        """
        self.dir = dir
        self.tstRatio = tstRatio
        self.batch_size = batch_size

        split_test_train_data(dir, tstRatio)

        trnTransform = data_transforms['train'] 
        self.trainSet = DroneDataset(train=True, transform = trnTransform)

        tstTransform = data_transforms['val']
        self.testSet = DroneDataset(train= False, transform = tstTransform)

        self.dataloaders = {'train': torch.utils.data.DataLoader(self.trainSet, batch_size= batch_size, 
                                                            shuffle=True, num_workers=4),
                    'val': torch.utils.data.DataLoader(self.testSet, batch_size= batch_size,
                                                shuffle=True, num_workers=4)}

        self.dataset_sizes = {'train': len(self.trainSet),
                        'val':len(self.testSet)}

        self.class_names = self.trainSet.classes
        #print(self.class_names)

    def show_batch(self, save_as="sample.jpg"):
        """Show five sample images for verification of dataloaders.

        Get item internal fuction
        """
        # Get a batch of training data
        inputs, classes = next(iter(self.dataloaders['train']))
        images = denormalize(inputs,mean=(0.5404, 0.5918, 0.6219),std=(0.2771, 0.2576, 0.2998)).cpu().numpy()

        counter=0
        fig = plt.figure(figsize=(15, 10))

        while(counter<5):
            ax = fig.add_subplot(1, 5, counter+1, xticks=[], yticks=[])
            img = images[counter]
            npimg = np.clip(np.transpose(img,(1,2,0)), 0, 1)
            ax.imshow(npimg, cmap='gray')
            ax.set_title(f'{self.class_names[classes[counter]]}', color= "blue",fontsize=16)
            counter+=1
        fig.tight_layout()  
        plt.show()
        fig.savefig(save_as)
        #imshow_save(out, save_as="sample.jpg",title=[class_names[int(x)] for x in classes[0:4]])

