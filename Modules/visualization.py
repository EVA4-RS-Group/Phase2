'''Plotting Utility.

Grad-CAM implementation in Pytorch

Reference:
[1] xyz
[2] xyz
'''

import matplotlib.pyplot as plt
import numpy as np
import torch

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

def visualize_model(model, data, device, save_as="visualize.jpg"):
    dataloaders, class_names = data.dataloaders, data.class_names
    was_training = model.training
    model.eval()
    images_so_far = 0
    figure = plt.figure(figsize=(15, 10))
    num_images=5

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            inputs = denormalize(inputs,mean=(0.5404, 0.5918, 0.6219),std=(0.2771, 0.2576, 0.2998)).cpu().numpy()

            for j in range(inputs.shape[0]):
                images_so_far += 1
                
                img = inputs[j]
                npimg = np.clip(np.transpose(img,(1,2,0)), 0, 1)
                ax = figure.add_subplot(1, 5, images_so_far, xticks=[], yticks=[])
                ax.imshow(npimg, cmap='gray')
                ax.set_title('predicted:\n{}'.format(class_names[preds[j]]),fontsize=14)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    figure.savefig(save_as)
                    return
        model.train(mode=was_training)
    figure.tight_layout()  
    plt.show()

def visualize_face_recog_model(model, data, device, save_as="visualize.jpg"):
    dataloaders, class_names = data.dataloaders, data.class_names
    was_training = model.training
    model.eval()
    images_so_far = 0
    figure = plt.figure(figsize=(15, 12))
    num_images=35

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            inputs = denormalize(inputs,mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)).cpu().numpy()

            for j in range(inputs.shape[0]):
                images_so_far += 1
                
                img = inputs[j]
                npimg = np.clip(np.transpose(img,(1,2,0)), 0, 1)
                ax = figure.add_subplot(5, 7, images_so_far, xticks=[], yticks=[])
                ax.imshow(npimg, cmap='gray')
                ax.set_title('{}'.format(class_names[preds[j]]),fontsize=12)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    figure.savefig(save_as)
                    return
        model.train(mode=was_training)
    figure.tight_layout()  
    #plt.title("Predicted Label",fontsize=16)
    figure.suptitle("Predicted Label",fontsize=16)
    figure.subplots_adjust(top=0.88)
    plt.show()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5404, 0.5918, 0.6219])
    std = np.array([0.2771, 0.2576, 0.2998])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title[:4])
    plt.pause(0.001)  # pause a bit so that plots are updated

def imshow_save(inp, save_as="sample.jpg",title=None):
    
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5404, 0.5918, 0.6219])
    std = np.array([0.2771, 0.2576, 0.2998])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    figure = plt.figure()
    plt.imshow(inp)
    if title is not None:
        plt.title(title[:4])
    plt.pause(0.001)  # pause a bit so that plots are updated
    figure.savefig(save_as)



def PlotGraph(plotData,save_as):
    fig, (axs1,axs2) = plt.subplots(2, 1,figsize=(15,10))
    axs1.plot(plotData['trainLoss'], label = " Train")
    axs1.plot(plotData['valLoss'], label = " Test")
    axs1.set_title("Loss", fontsize=16)

    axs2.plot(plotData['trainAccu'], label = " Train")
    axs2.plot(plotData['valAccu'], label = " Test")
    axs2.set_title("Accuracy", fontsize=16)

    axs1.legend(fontsize=14)
    axs2.legend(fontsize=14)
    axs1.tick_params(axis='both', which='major', labelsize=12)
    axs2.tick_params(axis='both', which='major', labelsize=12)
    plt.show()
    fig.savefig(save_as)