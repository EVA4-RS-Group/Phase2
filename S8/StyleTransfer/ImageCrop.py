import cv2


def cropeImage(image1, image2):
	img1 = cv2.imread(image1)
	img2 = cv2.imread(image2)
	print("img1 ", img1.shape)
	print("img2 ", img2.shape)
	rw1, col1, chnl1 = img1.shape
	rw2, col2, chnl2 = img2.shape

	if((rw1<=rw2 and col1<col2) or (rw1<rw2 and col1<=col2)):
		difRow = rw2 - rw1
		difCol = col2 - col1
		img2 = img2[int(difRow/2): rw2-int(difRow/2), int(difCol/2) : col2-int(difCol/2)]
		print("1")
	elif((rw2<=rw1 and col2<col1) or (rw2<rw1 and col2<=col1)):
		difRow = rw1 - rw2
		difCol = col1 - col2
		img1 = img1[int(difRow/2): rw1-int(difRow/2), int(difCol/2) : col1-int(difCol/2)]
		print("2")
	
	elif((rw1<=rw2 and col2<col1) or (rw1<rw2 and col2<=col1)):
		difRow = rw2 - rw1
		difCol = col1 - col2
		img2 = img2[int(difRow/2): rw2-int(difRow/2), 0: col2]
		img1 = img1[0: rw1, int(difCol/2) : col1-int(difCol/2)]
		print("3")

	elif((rw2<=rw1 and col1<col2) or (rw2<rw1 and col1<=col2)):
		difRow = rw1 - rw2
		difCol = col2 - col1
		img2 = img2[0: rw2, int(difCol/2) : col2-int(difCol/2)]
		img1 = img1[int(difRow/2) : rw1-int(difRow/2), 0 : col1]
		print("4")
	else:
		pass
	print("img1 ", img1.shape)
    print("img2 ", img2.shape)
    return img1, img2


#img1 = cv2.imread("mn.jpg")
#img2 = cv2.imread("me.jpg")

#cropeImage(img1,img2)
