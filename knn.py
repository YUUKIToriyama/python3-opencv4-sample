#!/usr/bin/python3

import cv2 as cv
import sys

def feature_point_matching():
	files = sys.argv
	
	img1 = cv.imread(files[1])
	img2 = cv.imread(files[2])
	
	akaze = cv.AKAZE_create()
	
	kp1, des1 = akaze.detectAndCompute(img1, None)
	kp2, des2 = akaze.detectAndCompute(img2, None)
	
	bf = cv.BFMatcher()
	
	matches = bf.knnMatch(des1, des2, k=2)
	
	ratio = 0.1
	good = []
	for m,n in matches:
	    if m.distance < ratio * n.distance:
	        good.append([m])
	
	img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
	
	cv.imshow("img", img3)
	
	cv.waitKey(0)
	cv.destroyAllWindows()

if __name__ == "__main__":
    feature_point_matching()
