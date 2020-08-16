import sys, cv2,dlib, time
import numpy as np
import faceblendcommon as fbc
import matplotlib.pyplot as plt

'''
File required :
https://github.com/EVA4-RS-Group/Phase2/blob/master/S3_FaceAlignment/faceblendcommon.py
Dlib Requirements:
  !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  !bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2

isValidImage Requirements:
 !wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

faceMask Requirements
Mask image : https://github.com/EVA4-RS-Group/Phase2/releases/download/s2/3M-KN95-9501-Dust-Mask_v1.jpg

Image Alignment:
!wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
!bzip2 -dk shape_predictor_5_face_landmarks.dat.bz2
'''

def isValidImage(image):
  #image = cv2.imread(imagePath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
  faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.3,
      minNeighbors=3,
      minSize=(70, 70)
  )
  if (len(faces) == 1):
    return True
  else:
    return False 


def faceSwap(img1, img2):
  img1Warped = np.copy(img2)
  im1Display = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
  im2Display = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

  points1 = fbc.getLandmarks(detector, predictor, img1)
  points2 = fbc.getLandmarks(detector, predictor, img2)

  # Find convex hull
  hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

  # Create convex hull lists
  hull1 = []
  hull2 = []
  for i in range(0, len(hullIndex)):
      hull1.append(points1[hullIndex[i][0]])
      hull2.append(points2[hullIndex[i][0]])

  # Calculate Mask for Seamless cloning
  hull8U = []
  for i in range(0, len(hull2)):
      hull8U.append((hull2[i][0], hull2[i][1]))

  mask = np.zeros(img2.shape, dtype=img2.dtype) 
  cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

  # Find Centroid
  m = cv2.moments(mask[:,:,1])
  center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))

  # Find Delaunay traingulation for convex hull points
  sizeImg2 = img2.shape    
  rect = (0, 0, sizeImg2[1], sizeImg2[0])

  dt = fbc.calculateDelaunayTriangles(rect, hull2)

  # If no Delaunay Triangles were found, quit
  if len(dt) == 0:
      quit()


  imTemp1 = im1Display.copy()
  imTemp2 = im2Display.copy()

  tris1 = []
  tris2 = []
  for i in range(0, len(dt)):
      tri1 = []
      tri2 = []
      for j in range(0, 3):
          tri1.append(hull1[dt[i][j]])
          tri2.append(hull2[dt[i][j]])

      tris1.append(tri1)
      tris2.append(tri2)

  # Simple Alpha Blending
  # Apply affine transformation to Delaunay triangles
  for i in range(0, len(tris1)):
      fbc.warpTriangle(img1, img1Warped, tris1[i], tris2[i])

  # Clone seamlessly.
  output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
  return output[:,:,::-1]



def testfaceSwap():
  img2 = cv2.imread('/content/rahul-gandhi.jpg')
  img1 = cv2.imread('/content/Hrithik-Roshan.jpg')
  if(isValidImage(img1) and isValidImage(img2)):
    plt.figure(figsize=(20,10))
    plt.imshow(faceSwap(img1, img2))
    plt.axis('off')


#############################################################################################
#                 FaceMask
###########################################################################################
def faceMask(face):    # Read images
    maskImg = cv2.imread('3M-KN95-9501-Dust-Mask_v1.jpg') # Mask image

    im1Display = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    im3Display = cv2.cvtColor(maskImg, cv2.COLOR_BGR2RGB)

    img1Warped = np.copy(img1)

    # Initialize the dlib facial landmakr detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # Read array of corresponding points
    points1 = fbc.getLandmarks(detector, predictor, img1)
    points3 = fbc.getLandmarks(detector, predictor, maskImg)

    points3_new = points3[2:16]

    # Create convex hull lists
    hull1 = points1[1:16]
    hull3 = points3[1:16]

    # Calculate Mask for Seamless cloning
    hull8U = []
    for i in range(0, len(hull1)):
        hull8U.append((hull1[i][0], hull1[i][1]))

    mask = np.zeros(img1.shape, dtype=img1.dtype) 
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    # Find Centroid
    m = cv2.moments(mask[:,:,1])
    center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))

    # Find Delaunay traingulation for convex hull points
    sizeImg1 = img1.shape    
    rect = (0, 0, sizeImg1[1], sizeImg1[0])

    dt = fbc.calculateDelaunayTriangles(rect, hull1)

    # If no Delaunay Triangles were found, quit
    if len(dt) == 0:
        quit()

    imTemp1 = im1Display.copy()
    imTemp3 = im3Display.copy()

    tris1 = []
    tris3 = []
    for i in range(0, len(dt)):
        tri1 = []
        tri3 = []
        for j in range(0, 3):
            tri1.append(hull1[dt[i][j]])
            tri3.append(hull3[dt[i][j]])

        tris1.append(tri1)
        tris3.append(tri3)

    # Simple Alpha Blending
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(tris3)):
        fbc.warpTriangle(maskImg, img1Warped, tris3[i], tris1[i])

    # Calculate Mask for Seamless cloning
    hull8U = []
    for i in range(0, len(hull1)):
        hull8U.append((hull1[i][0], hull1[i][1]))

    mask = np.zeros(img1.shape, dtype=img1.dtype) 
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    # Find Centroid
    m = cv2.moments(mask[:,:,1])
    center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img1, mask, center, cv2.NORMAL_CLONE)
    return np.uint8(img1Warped)[:,:,::-1], output[:,:,::-1]

def TestFaceMask():
  img1 = cv2.imread('/content/arvind.png')
  if(isValidImage(img1)):
    plt.figure(figsize=(20,10))
    plt.subplot((141))
    plt.imshow(faceMask(img1)[0])
    plt.axis('off');
    plt.subplot((142))
    plt.imshow(faceMask(img1)[1])
    plt.axis('off');

    
#################################################################
#               Image alignment 
#################################################################
def alignImage(image):
    # Landmark model location
    MODELPATH = "/content/model/"
    PREDICTOR_PATH = MODELPATH + "shape_predictor_5_face_landmarks.dat"
    # Get the face detector
    faceDetector = dlib.get_frontal_face_detector()
    # The landmark detector is implemented in the shape_predictor class
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
    # Read image
    # Detect landmarks
    points = fbc.getLandmarks(faceDetector, landmarkDetector, image)
    points = np.array(points)
    # Convert image to floating point in the range 0 to 1
    image = np.float32(image)/255.0
    # Dimension of output image
    h = 600
    w = 600
    # Normalize image to output co-orindates
    imNorm, points = fbc.normalizeImagesAndLandmarks((h,w), image, points)
    imNorm = np.uint8(imNorm*255)
    return imNorm[:,:,::-1]

def testImgAlignment():
    im = cv2.imread("/content/data/images/face2.jpg")
    plt.imshow(alignImage(im))
    plt.title("Aligned Image")
    plt.axis("off")
    plt.show()
