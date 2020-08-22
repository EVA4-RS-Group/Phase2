import matplotlib
import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt
import torch


def ShowMissclassifiedImages(model, data, class_id, device,dataType='val', num_images=12,save_as="misclassified.jpg"):
    dataloaders, class_names = data.dataloaders, data.class_names
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig, axs = plt.subplots(int(num_images/4),4,figsize=(12,12))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[dataType]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
              
            for j in range(inputs.size()[0]):
                if((preds[j] != labels[j]) and (labels[j] == class_id)):
                  row = int((images_so_far)/4)
                  col = (images_so_far)%4
                  imagex = inputs.cpu().data[j]
                  imagex = np.transpose(imagex, (1, 2, 0))
                  imagex=imagex.numpy()
                  mean = np.array([0.53713346, 0.58979464, 0.62127595])
                  std = np.array([0.27420551, 0.25534403, 0.29759673])
                  imagex = std*imagex  + mean
                  imagex = np.clip(imagex, 0, 1)       
                  axs[row,col].imshow(imagex)
                  axs[row,col].axis('off')
                  fig.tight_layout(pad=2.0)
                  axs[row,col].set_title('Predicted: {} \n Actual: {}'.format(class_names[preds[j]],class_names[labels[j]]))
                  images_so_far += 1
                  if images_so_far == num_images:
                      model.train(mode=was_training)
                      plt.show()
                      fig.savefig(save_as)
                      return
        model.train(mode=was_training)

def ShowCustomDataFaces_plot(model, data, class_id, device,dataType='val', num_images=6,save_as="misclassified.jpg"):
    dataloaders, class_names = data.dataloaders, data.class_names
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(8, 4))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[dataType]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
              
            for j in range(inputs.size()[0]):
                if labels[j] == class_id:
                  row = 0
                  col = images_so_far+1
                  imagex = inputs.cpu().data[j]
                  imagex = np.transpose(imagex, (1, 2, 0))
                  imagex=imagex.numpy()
                  mean = np.array([0.485, 0.456, 0.406])
                  std = np.array([0.229, 0.224, 0.225])
                  imagex = std*imagex  + mean
                  imagex = np.clip(imagex, 0, 1)  
                  ax = fig.add_subplot(1, 6, col, xticks=[], yticks=[])     
                  ax.imshow(imagex)
                  ax.set_title('{}'.format(class_names[preds[j]]))
                  images_so_far += 1
                  if images_so_far == num_images:
                      model.train(mode=was_training)
                      fig.tight_layout()
                      plt.show()
                      fig.savefig(save_as)
                      return
        model.train(mode=was_training)

def ShowCustomDataFaces(model, data, device,dataType='val', num_images=6):
    print("------------  Prediction on Validation set of Images for Aishwarya Rai -------------------")
    ShowCustomDataFaces_plot(model, data,0, device,dataType,save_as="Predictions_AishwaryaRai.jpg")
    print("------------  Prediction on Validation set of Images for Elon Musk -------------------")
    ShowCustomDataFaces_plot(model, data,14, device,dataType,save_as="Predictions_ElonMusk.jpg")
    print("------------  Prediction on Validation set of Images for Mahendra Singh Dhoni -------------------")
    ShowCustomDataFaces_plot(model, data,43, device,dataType,save_as="Predictions_MahendraSinghDhoni.jpg")
    print("------------  Prediction on Validation set of Images for Malala Yousafzai -------------------")
    ShowCustomDataFaces_plot(model, data,45, device,dataType,save_as="Predictions_MalalaYousafzai.jpg")
    print("------------  Prediction on Validation set of Images for Narendra Modi -------------------")
    ShowCustomDataFaces_plot(model, data,49, device,dataType,save_as="Predictions_NarendraModi.jpg")
    print("------------  Prediction on Validation set of Images for Priyanka Chopra -------------------")
    ShowCustomDataFaces_plot(model, data,53, device,dataType,save_as="Predictions_PriyankaChopra.jpg")
    print("------------  Prediction on Validation set of Images for Rahul Gandhi -------------------")
    ShowCustomDataFaces_plot(model, data,54, device,dataType,save_as="Predictions_RahulGandhi.jpg")
    print("------------  Prediction on Validation set of Images for Sachin Tendulkar -------------------")
    ShowCustomDataFaces_plot(model, data,59, device,dataType,save_as="Predictions_SachinTendulkar.jpg")
    print("------------  Prediction on Validation set of Images for Shahrukh Khan -------------------")
    ShowCustomDataFaces_plot(model, data,62, device,dataType,save_as="Predictions_ShahrukhKhan.jpg")
    print("------------  Prediction on Validation set of Images for Shreya Ghoshal -------------------")
    ShowCustomDataFaces_plot(model, data,63, device,dataType,save_as="Predictions_ShreyaGhoshal.jpg")