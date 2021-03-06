# Session 2 - Deploying Mobilenet_v2 model for Custom Dataset


## 1. Executive Summary
**Group Members:** *Ramjee Ganti, Srinivasan G, Roshan, Dr. Rajesh and Sujit Ojha* 

### **Objectives**:

- Group challenge involving data curation for different flying objects like bird, drones. Our group target is to curate 1000 images of large quadcopter.
- Train mobilenet_v2 model for custom dataset
- Deploy the model on AWS Lambda using Serverless and then use it to classify our images. 

### **Results**:
- Team added 1020 large quadcopter images
- MobileNetv2 pre-trained model with one-cycle policy:   
https://github.com/EVA4-RS-Group/Phase2/blob/master/S2_mobilenet_v2_custom_dataset/Training/EVA4_S2_mobilnet_v2_transfer_learning_v6B.ipynb
    - Training accuracy : 96.53%
    - Test accuracy: 95.61% 
- Deployed the model on AWS Lambda: https://3g8t28a24d.execute-api.ap-south-1.amazonaws.com/dev/classify

### **Details**:
- Resizing strategy
    - Resized all images above 448 pixel to 448 keeping aspect ratio same
    - Used RandomResizedCrop in training to manage the different size and got it to 224x224 pixel. It had effect like cutout to control overfitting.
    - Deployment: we resize the incoming images to 256 keeping aspect ratio same and then we did center crop transformation.
- Developer Section includes details about 
    - Code explaination
    - What model did we trained? MobileNetv2 pre-trained using one cycle policy
    - Accuracy vs epochs graphs for train and test curves
    - Accuracy vs epochs graphs for train and test curves are also include in developer section
    - 10 misclassified images for each of the classes as an Image Gallery



### **Key Highlights**
- Data curation issues 
    - Some of the teams didn't followed the requirements.
    - Eg. Flying birds had many issues like, large count of flocking birds, cartoon, non-bird 
    - Most of the small quadropter are big size then 1 ft.
- Data/annotation cleaning: Team reviewed the mis-classification images and manually reviewed/updated the annotation.
- One Cycle policy learning in two stages gave us state of art accuracies considering still annotation errors are present
- Deployment challenges with torch version in colab and AWS.
    - Recently colab upgraded torch to version 1.6 and if we zip it, it create total package size of 167mb (>150mb)
    - We specified the torch version same in colab to resolve this issue.


## 2. Steps (Developer Section)

1. Data Curation:
     - Curated dataset from Kaggle, google, youtube video meeting all the requirements.

2. Data preprocessing & Cleanup:
    - Image with size > 448 pixel is resized to 448 pixel, keeping aspect ratio same.
    - Images converted to *.jpg
    - Removed the .ipynb and txt
    - Removed missplaced images from each class in the dataset.
    - Renamed the files into consistent naming convention and mapping file for classes. 
    - Curated dataset link: https://drive.google.com/drive/folders/1YNjGPnE-yft5HRgKb01cQEWvrEUQci_V

3. Data Structure:  After clean-up, the images collected where placed in to 4 folders wthin zip file with root as final_images_ver2
    - Flying Birds  
    - Large Quadcopters   
    - Small Quadcopters   
    - Winged Drones   

4. All images zipped present at  [final_images_ver2.zip](https://github.com/EVA4-RS-Group/Phase2/releases/download/s2/final_images_ver2.zip)
    - **datset.csv**  - contains list of images and the corresponding class name. Created by program present in test_train_csvGen.py
    - **img_class.csv** - contains list of all images and their respective classes. Created by program present in test_train_csvGen.py

5. Calculate mean and standard deviation for the dataset [find_mean_stddev.py](https://github.com/EVA4-RS-Group/Phase2/blob/master/Modules/find_mean_stddev.py)
    - Calculated mean and standard deviation across all images. 
    
        **Mean :** [0.53614524, 0.58885578, 0.62041911]
        
        **Std  :**  [0.27362225, 0.25486343, 0.2974289 ]
    - This gets used in data tranforms
     
6. Create DroneDataSet at [data_loader.py](https://github.com/EVA4-RS-Group/Phase2/blob/master/Modules/data_loader.py)
   DroneSet class that inherits from torch.utils.data.Dataset and perform following
    - Takes as input : train parameter, class names and transforms
    - Load image data based on value of boolean flag train , train data or test data
    - Load Class names
    - Perform transforms on images data 
    - Visualize images
    
                                               Sample Images
![Image of Yaktocat](https://github.com/EVA4-RS-Group/Phase2/blob/master/S2_mobilenet_v2_custom_dataset/Training/output/sample.jpg)
7. Prepare data for training and testing:
    - Split image data as  train data and test data
    - Store images respectively in trainData.csv and testData.csv
    - Based on input parameter passed to DroneSet, the corresponding data gets loads

8. Load **mobilenet_v2** model and determine lr hyper parameter that is passed as input to optimizer and scheduler used in training:
    - Load  model with **pertained option = true**
    - Set value of model parameters **requires_grad = False**
    - Reconstruct final fully connected layer. 
    -   - Parameters of **newly constructed modules have requires_grad=True by default**
    - Pass model to LRFinder(torch-lr-finder)
    - Plot LR values reported by LRFinder
    - Run range_test to determine lr value  using  train dataset

9. Setup Optimizer as **SGD optimiser** and Scheduler as **OneCycleLR**
    - Only parameters of final layer are being optimized 
    - Using LR find, we got the max LR for OneCyclePolicy and created optimizer with default parameters for 25 epoch.

10. Train by train_model function present in [train.py](https://github.com/EVA4-RS-Group/Phase2/blob/master/Modules/train.py) for 25 epochs:
    - Returns model, loss and accuracy
    - Plot Loss and Accuracy for the train model
    - Visualize images
![Image of Yaktocat](https://github.com/EVA4-RS-Group/Phase2/blob/master/S2_mobilenet_v2_custom_dataset/Training/output/loss_accuracy_1.jpg)

                                              Sample Prediction
![Image of Yaktocat](https://github.com/EVA4-RS-Group/Phase2/blob/master/S2_mobilenet_v2_custom_dataset/Training/output/visualize_1.jpg)

11. Load **mobilenet_v2** model:
    - Set value of model parameters **requires_grad = False**
    - Pass model to LRFinder(torch-lr-finder)
    - Plot LR values reported by LRFinder
    - Run range_test to determine lr value  using  train dataset

12. Train by train_model function present in [train.py](https://github.com/EVA4-RS-Group/Phase2/blob/master/Modules/train.py) for 25 epochs
    - Returns model, loss and accuracy
    - Plot Loss and Accuracy for the train model
    - Visualize images
![Image of Yaktocat](https://github.com/EVA4-RS-Group/Phase2/blob/master/S2_mobilenet_v2_custom_dataset/Training/output/loss_accuracy_2.jpg)

                                            Sample Prediction
![Image of Yaktocat](https://github.com/EVA4-RS-Group/Phase2/blob/master/S2_mobilenet_v2_custom_dataset/Training/output/visualize_2.jpg)

13. Find and Display misclassified images from each classes
    - Find misclassified images
    - Display misclassified images
    
    
                                                Misclassified Flying Birds
![Image of Yaktocat](https://github.com/EVA4-RS-Group/Phase2/blob/master/S2_mobilenet_v2_custom_dataset/Training/output/MisclassifiedFlyingBirds.jpg)
                                    
                                                Misclassified Large Quadcopter
![Image of Yaktocat](https://github.com/EVA4-RS-Group/Phase2/blob/master/S2_mobilenet_v2_custom_dataset/Training/output/MisclassifiedLargeQuadcopter.jpg)
                                     
                                                Misclassified Small Quadcopter
![Image of Yaktocat](https://github.com/EVA4-RS-Group/Phase2/blob/master/S2_mobilenet_v2_custom_dataset/Training/output/MisclassifiedSmallQuadcopter.jpg)
                                     
                                                Misclassified Winged Drones
![Image of Yaktocat](https://github.com/EVA4-RS-Group/Phase2/blob/master/S2_mobilenet_v2_custom_dataset/Training/output/MisclassifiedWingedDrones.jpg)
 
14. Save model
    - Save model
    - Please confirm the saved model can be loaded and evaludated again.

## 3. References

1. [Image utils developed by Ramjee Ganti](https://github.com/gantir/image_utils/blob/master/utils.py)
