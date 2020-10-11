# Session 7 - Super Resolution GAN


## 1. Executive Summary
**Group Members:** *Ramjee Ganti, Srinivasan G, Roshan, Dr. Rajesh and Sujit Ojha*

### **Objectives**:

- Implement Variational Super resolution GAN. Read the [paper](https://arxiv.org/pdf/1609.04802.pdf) 
- Upload the model to AWS lambda
- Make sure that the model generate super resolution image.

### **Results**:

- Team hosted static website : http://rsgroup.s3-website.ap-south-1.amazonaws.com/
- Website results
    - <img src="results/website_results1.png" alt="Set1" height="300"/><img src="results/website_results2.png" alt="set2" height="300"/>
- Colab results
 **Sample image and output**

    **Low Resolution image**
    - <img src="results/LR_img.png" alt="Low resolution image input" height="300"/>
    
    **High Resolution image**
    - <img src="results/HR_image.png" alt="High Resolution image" height="300"/>
    
     **Fake High Resolution image**
    - <img src="results/fakeHR_img.png" alt="Fake High Resolution image" height="300"/>
    
 **On Text image**
 
    **Low Resolution image**
    - <img src="results/LR_img2.png" alt="Low resolution image input" height="300"/>
    
    **High Resolution image**
    - <img src="results/HR_image2.png" alt="High Resolution image" height="300"/>
    
     **Fake High Resolution image**
    - <img src="results/fakeHR_img2.png" alt="Fake High Resolution image" height="300"/>


### **Image Super Resolution**

![Image](https://github.com/EVA4-RS-Group/Phase2/blob/master/S8/SR_GAN/results/SRGAN.png)

- Super-resolution GAN applies a deep network in combination with an adversary network to produce higher resolution images.   
- During the training, A high-resolution image (HR) is downsampled to a low-resolution image (LR). 
  A GAN generator upsamples LR images to super-resolution images (SR). 
  We use a discriminator to distinguish the HR images and backpropagate the GAN loss to train the discriminator and the generator.
- The propposed model is capable of generating image of 4X sinze.
- Adam optimizer is used in both generator and descrimantor network.
- Training is done for 200 epochs.


## 2. Steps (Developer Section)

## 3. References

1. [EVA4 Phase2 Session7, Variational Auto Encoder](https://theschoolof.ai/)
2. [SRGAN implementation] (https://github.com/leftthomas/SRGAN)
3. [SRGAN Paper] (https://arxiv.org/pdf/1609.04802.pdf)

