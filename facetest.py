#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import dlib
import cv2
import os
import time
import numpy as np

def getCapVideo():	
	# 获取视频流
	cap = cv2.VideoCapture(0)
	# 获取检测器
	detector = dlib.get_frontal_face_detector() #获取人脸分类器
	predictor = dlib.shape_predictor('./cv_source/shape_predictor_68_face_landmarks.dat')
	while cap.isOpened():
		ret,frame = cap.read()
		# 分离三个颜色通道
		# b, g, r = cv2.split(img) 
		# b = frame[:,:,0]
		# g = frame[:,:,1]
		# r = frame[:,:,2]   
		# print time.time()
		# # 融合三个颜色通道生成新图片
		# img2 = cv2.merge([r, g, b])
		img2 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
		dets = detector(frame, 1) #使用detector进行人脸检测 dets为返回的结果
		for index, face in enumerate(dets):
			shape = predictor(frame, face)  # 寻找人脸的68个标定点 
			# 遍历所有点，打印出其坐标，并用蓝色的圈表示出来
			for index, pt in enumerate(shape.parts()):
				# print 'Part {}: {}'.format(index, pt)
				pt_pos = (pt.x, pt.y)
				cv2.circle(frame, pt_pos, 2, (255, 0, 0), 3)
		
		# 在新窗口中显示
		cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('image', frame)

		# 等待按键，随后退出，销毁窗口
		k = cv2.waitKey(1) & 0xFF
		if k == ord('q'):break

	cap.release()
	cv2.destroyAllWindows()

# 参数是照片
def deteFace(img):
	#os.getcwd()  # 获取当前路径
	predictor_path = './cv_source/shape_predictor_68_face_landmarks.dat'
	detector = dlib.get_frontal_face_detector() #获取人脸分类器
	predictor = dlib.shape_predictor(predictor_path)    # 获取人脸检测器

	# 摘自官方文档：
	# image is a numpy ndarray containing either an 8bit grayscale or RGB image.
	# opencv读入的图片默认是bgr格式，我们需要将其转换为rgb格式；都是numpy的ndarray类。
	# b, g, r = cv2.split(img)    # 分离三个颜色通道
	b = img[:,:,0]
	g = img[:,:,1]
	r = img[:,:,2]
	# print r,b,g
	img2 = cv2.merge([r, g, b])   # 融合三个颜色通道生成新图片

	dets = detector(img, 1) #使用detector进行人脸检测 dets为返回的结果
	# print "Number of faces detected: {}".format(len(dets))   # 打印识别到的人脸个数
	# enumerate是一个Python的内置方法，用于遍历索引
	# index是序号；face是dets中取出的dlib.rectangle类的对象，包含了人脸的区域等信息
	# left()、top()、right()、bottom()都是dlib.rectangle类的方法，对应矩形四条边的位置
	
	for index, face in enumerate(dets):
		# print 'face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom())
		shape = predictor(img, face)  # 寻找人脸的68个标定点 

		# print(shape)
		# print(shape.num_parts)
		# 遍历所有点，打印出其坐标，并用蓝色的圈表示出来
		for index, pt in enumerate(shape.parts()):
			# print 'Part {}: {}'.format(index, pt)
			pt_pos = (pt.x, pt.y)
			cv2.circle(img, pt_pos, 2, (255, 0, 0), 2)

		# 在新窗口中显示
		cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
		cv2.imshow('image', img)

		# 等待按键，随后退出，销毁窗口
		k = cv2.waitKey(0) & 0xFF
		if k == ord('q'):break
	cv2.destroyAllWindows()

if __name__ == '__main__':
	getCapVideo()
	# opencv 读取图片，并显示
	# img = cv2.imread('./cv_source/testimg.jpeg', cv2.IMREAD_COLOR)
	# deteFace(img)









