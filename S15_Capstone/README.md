# Session 14 - Capstone Project


## 1. Executive Summary
**Group Members:** *Ramjee Ganti, Srinivasan G, Roshan, Dr. Rajesh and Sujit Ojha*

### **Objectives**:

Develop a [lobe.ai](https://lobe.ai/) clone:
    1. allow people to upload their own data:
        a. Images (10x2 - 80:20) upto (100*10 - 70-30)
        b. text samples (csv files) (min 100 - max 1000)*2-3 - 70:30)
    2. plug-in a model of your choice
    3. train the model on AWS EC2, use lambda to preprocess, trigger EC2 and match user id with the files stored on EC2, and for inferencing (as discussed in the class, refer the video below)
    4. move the model to lambda and show inferencing

Points to be noted:

    1. If you are doing object detection, you need to build a front-end for annotation, and then you do not need to do 1.b.
    2. you need to trigger the AWS training using lambda, so it is not always on
    3. try and pre-process the data on lambda or in browser to make sure you are not uploading BIG data
    4. limit max images per class to 100 and max classes to 10
    5. use transfer learning


## 4. References

1. [EVA4 Phase2 Session 15 Capstone](https://theschoolof.ai/)
2. [lobe.ai, A product by Microsoft](https://lobe.ai/)

