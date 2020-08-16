# Session 3 - Face Recognition & AWS static Website


## 1. Executive Summary
**Group Members:** *Ramjee Ganti, Srinivasan G, Roshan, Dr. Rajesh and Sujit Ojha* 

### **Objectives**:

- Upload html and js files to your S3 bucket and create a policy using which the html file can be accessed by anyone. The HTML file should contain:
    1. ResNet Example (as shared in the code above)
    2. MobileNet Example (trained on your dataset)
    3. Face Alignment Feature (as shared above)
        - Bonus 1000 points additional from 3000 for this assignment if you implement Face Swap. 
- Create a Face Alignment App on Lambda (code is shared above), where if someone uploads a face (you check that by using dlib face detector), you return aligned face. Image with more than 1 face is not processed for alignment. 
- Share the link to your S3 html file that can be accessed by anyone. Also share the link to your GitHub repo for the code (please remember to always remove the keys, secrect_keys, etc from your code before uploading to GitHub. How?)

### **Results**:
- Team hosted static website : http://rsgroup.s3-website.ap-south-1.amazonaws.com/
    - Resnet imagenet from [Session 1](https://github.com/EVA4-RS-Group/Phase2/tree/master/S1_aws_lambda_mobilenet_v2)
    - MobileNet Example from [Session 2](https://github.com/EVA4-RS-Group/Phase2/tree/master/S2_mobilenet_v2_custom_dataset)
    - FaceAlignment & FaceSwap application - putting mask, Details are covered in Developer Section.
- Website results (Week 1)
    - <img src="results/week1.png" alt="Week1" width="400"/>


### **Key Highlights**
- Website hosting (#Ramjee, Please highlight key things in website hos)
- Face Alignment (#Roshan, please highlight)
- Face Swap, Mask 


## 2. Steps (Developer Section)
- Website Hosting (#Ramjee, Detailed steps)
- Face Alignment (#Roshan, Detailed Steps)
- Face Swap, Mask [EVA4_Phase2_S3_FaceMask.ipynb](EVA4_Phase2_S3_FaceMask.ipynb)


## 3. References

1. [Hosting AWS static website](https://docs.aws.amazon.com/AmazonS3/latest/dev/HostingWebsiteOnS3Setup.html)
2. [EVA4 Phase2 Session3, Face Recognition Part 1](https://theschoolof.ai/)
