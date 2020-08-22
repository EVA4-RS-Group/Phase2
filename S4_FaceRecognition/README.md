# Session 4 - Face Recognition Part II, 10 Celebrities/Politician


## 1. Executive Summary
**Group Members:** *Ramjee Ganti, Srinivasan G, Roshan, Dr. Rajesh and Sujit Ojha*

### **Objectives**:

- Refer to this beautiful [blog](https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79). 
- Collect 10 facial images of 10 people you know (stars, politicians, etc). The more the images you collect, the better your experience would be. Add it to this [LFW](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz) dataset. 
- Train as in the blog and upload the FR model to Lambda
- Share the link to "that" single page. 
- Share the link to GitHub repo. 

### **Results**:
- Team hosted static website : http://rsgroup.s3-website.ap-south-1.amazonaws.com/
- Website results
    - <img src="results/week1.png" alt="Week1" height="200"/>


### **Key Highlights**
- Training Dataset Curation
- Face Alignment (#Roshan, please highlight)
- Face Swap, Mask: We developed a face swap application which will put mask on image with single face. It utilizes landmarks below eye portion to develop a convex hull. Then it swaps this portion with image having N95 mask.
- Deployment: Got two github actions one for [deploying the website](https://github.com/EVA4-RS-Group/Phase2/actions?query=workflow%3A%22Frontend+Deploy%22) and the other for [deploying to the lambda](https://github.com/gantir/eva4-2/actions?query=workflow%3A%22EVA4+Phase2+Week3%22). The actions get triggered when code is committed to master branch



## 2. Steps (Developer Section)
- Website Hosting
  - Used Pelican static site generator for python.
  - URL of website: http://rsgroup.s3-website.ap-south-1.amazonaws.com/
  - Created a theme based on [html5up](https://html5up.net/uploads/demos/phantom/)
  - Created 5 articles: Week1, Week2, Week3 FaceSwap, Week3 FaceMask, Week3 FaceAlign
  - Wrote javascript to upload image and use the same to call the Lambda inference API
- Face Alignment (#Roshan, Detailed Steps)
- Face Swap, Mask [EVA4_Phase2_S3_FaceMask.ipynb](EVA4_Phase2_S3_FaceMask.ipynb)
    - Using DLIB face detection and 68 points landmark detection
    - Face mask intermediate results
    - <img src="results/week3_face_mask_intermediate.png" alt="Week 3 Face align"/>


## 3. References

1. [Hosting AWS static website](https://docs.aws.amazon.com/AmazonS3/latest/dev/HostingWebsiteOnS3Setup.html)
2. [EVA4 Phase2 Session3, Face Recognition Part 1](https://theschoolof.ai/)
