#!/usr/bim/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv

def detect(filename):
	face_cascade = cv.CascadeClassifier('./cv_source/cascades/haarcascade_frontalface_default.xml')
	img = cv.imread(filename)
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.1, 6)
	print '人脸检测',faces

	for (x,y,w,h) in faces:
		img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
	cv.namedWindow('facedete')
	cv.imshow('facedete',img)
	cv.imwrite('./vidingdi.jpg',img)
	cv.waitKey(0)
	cv.destroyAllWindows()

if __name__ == '__main__':
	detect('/Users/junming/Desktop/timg.jpeg')
	


















