# Session 9 - Neural Embedding


## 1. Executive Summary
**Group Members:** *Ramjee Ganti, Srinivasan G, Roshan, Dr. Rajesh and Sujit Ojha*

### **Objectives**:

- Practise the above 6 colab files. 
- Move anyone to Lambda. 

### **Results**:

- Team hosted static website : http://rsgroup.s3-website.ap-south-1.amazonaws.com/
- Website results
    - <img src="results/website_snapshot_1.png" alt="Set1" height="300"/><img src="results/website_snapshot_2.png" alt="set2" height="300"/>
- Colab results
    - <img src="results/website_snapshot_1.png" alt="Set1" height="300"/><img src="website_snapshot_2.png" alt="Set2" height="300"/>


### **Key Highlights**
- Dataset curation & Preprocessing - [raw images](https://drive.google.com/drive/folders/1nskvo2QBLbtvIrXdoZeE5hRFp1WPNs3N?usp=sharing),  [processed images](https://github.com/EVA4-RS-Group/Phase2/releases/download/S6/processed_images_step4a.zip) and [EVA4_P2_S6_GenerativeAdversarialNetwork_Data_Preprocessing_v1.ipynb](Training/EVA4_P2_S6_GenerativeAdversarialNetwork_Data_Preprocessing_v1.ipynb)
- Training based on R1GAN Network[EVA4_P2_S6_R1GAN_128x128_2000_epoches_v1.ipynb](Training/EVA4_P2_S6_R1GAN_128x128_2000_epoches_v1.ipynb)
    - Updated the network to process 128x128 pixel instead of 64x64 pixels.
- Model Conversion & Inferencing[EVA4_P2_S6_R1GAN_128x128_2000_epoches_v1.ipynb](Training/EVA4_P2_S6_R1GAN_128x128_2000_epoches_v1.ipynb)
- Deployment in AWS Lambda using serverless and ONNX run time 


## 2. Steps (Developer Section)
- Dataset Curation & Preprocessing - [raw images](https://drive.google.com/drive/folders/1nskvo2QBLbtvIrXdoZeE5hRFp1WPNs3N?usp=sharing) and [processed images](https://github.com/EVA4-RS-Group/Phase2/releases/download/S6/processed_images_step4a.zip)
    - Downloaded 500+ images from https://www.cleanpng.com/, google and other care re-sale websites eg. cardekho.
    - Downloading criteria to keep the training simple,
        - White background or blank background
        - One direction of orientation or can be flipped to get same orientation
    - Processed images manually to flip the images and removed background for some of the images.
    - Finally processed all the image to generate 128x128 pixel images by resizing and padding white background. [EVA4_P2_S6_GenerativeAdversarialNetwork_Data_Preprocessing_v1.ipynb](Training/EVA4_P2_S6_GenerativeAdversarialNetwork_Data_Preprocessing_v1.ipynb)
- Training based on R1GAN, Reference #1 [EVA4_P2_S6_R1GAN_128x128_2000_epoches_v1.ipynb](Training/EVA4_P2_S6_R1GAN_128x128_2000_epoches_v1.ipynb)
    - Updated the Generator and Discriminator for 128x128 pixel with additional layer of convolution
    - Trained for 2000+ epoches to improve the images generated. Sample images below,
    - <img src="results/R1GAN_training_results_collage.png" alt="Indian Car Images generated" height="300"/><img src="results/R1GAN_training_results.png" alt="Indian Car image" height="300"/>
- Converted the model to onnx  [EVA4_P2_S6_R1GAN_128x128_onnx_v2.ipynb](Training/EVA4_P2_S6_R1GAN_128x128_onnx_v2.ipynb)
    - <img src="results/R1GAN_training_results_onnx_collage.png" alt="Indian Car Images generated" height="300"/><img src="results/R1GAN_training_results_onnx.png" alt="Indian Car image" height="300"/>
- Deployment [handler.py](GAN-Deployment/handler.py) and [serverless.yml](GAN-Deployment/serverless.yml)
    - Using serverless, python-plugin-requirements and onnxruntime


## 3. References

1. [PyTorch Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)
2. [Save and Load Machine Learning Models in Python with scikit-learn](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)
3. [How to Speed up AWS Lambda deployment on Serverless Framework by leveraging Lambda Layers.](https://gaurav4664.medium.com/how-to-speed-up-aws-lambda-deployment-on-serverless-framework-by-leveraging-lambda-layers-623f7c742af4)
4. [EVA4 Phase2 Session9, Neural Embedding](https://theschoolof.ai/)

