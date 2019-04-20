#!/usr/bim/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv

##################### 第四部分：CV中的图像处理 ##################### 
# 颜色空间转换
# 本小节目标
# 学习如何进行图像颜色空间转换，比如：BGR->灰度图，BGR->HSV
# 从特定图片展提取特定颜色物体
# 本节函数：cv2.cvtColor(),cv.inRange()等

# 使用hsv
def changeColorSpace():
	img = cv.imread('./cv_source/start.png')
	img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
	cv.namedWindow('image')
	cv.imshow('image',img_hsv)
	cv.waitKey(0)
	cv.destroyAllWindows()

# 函数：cv.inrange()。类似于threshold函数，实现二值化功能
# 函数原型：void inRange(InputArray src, InputArray lowerb,InputArray upperb, OutputArray dst);
# 参数1：输入要处理的图像，可以为单通道或多通道。
# 参数2：包含下边界的数组或标量。
# 参数3：包含上边界数组或标量。
# 参数4：输出图像，与输入图像src 尺寸相同且为CV_8U 类型
# 理解：类似于区间，某个像素数值在传入的参数区间内，变为255，之外变为0
def inrangeTest():
	img = cv.imread('./cv_source/IMG_4266.JPG')
	img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

	# 参数是：BGR
	uper_hsv = np.array([180,200,200])
	lower_hsv = np.array([24,39,42])
	mask = cv.inRange(img_hsv,lower_hsv,uper_hsv)
	
	# 做与运算
	res = cv.bitwise_and(img,img,mask=mask)

	cv.namedWindow('image')
	cv.imshow('image',res)
	cv.imwrite('./cv_source/new_mask.JPG',img_hsv)
	cv.imwrite('./cv_source/new_IMG_4266.JPG',res)
	cv.waitKey(0)
	cv.destroyAllWindows()

# 提取红黄蓝三种颜色的物体
def getBGRColor():
	img = cv.imread('./cv_source/opencv.jpeg')
	img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

	# 蓝色B范围
	lower_bule = np.array([110,100,100])
	upper_blue = np.array([130,255,255])
	mask_b = cv.inRange(img_hsv,lower_bule,upper_blue)
	res_b = cv.bitwise_and(img,img,mask=mask_b)

	# 红色R范围
	lower_red = np.array([0,100,100])
	upper_red = np.array([10,255,255])
	mask_r = cv.inRange(img_hsv,lower_red,upper_red)
	res_r = cv.bitwise_and(img,img,mask=mask_r)

	# 绿色G范围
	lower_green = np.array([60,100,100])
	upper_green = np.array([70,255,255])
	mask_g = cv.inRange(img_hsv,lower_green,upper_green)
	res_g = cv.bitwise_and(img,img,mask=mask_g)

	res = res_b+res_g+res_r

	cv.namedWindow('image')
	cv.imshow('image',res)
	cv.imwrite('./cv_source/new_mask.JPG',img_hsv)
	cv.imwrite('./cv_source/new_IMG_4266.JPG',res)
	cv.waitKey(0)
	cv.destroyAllWindows()


def getBlue():
	cap = cv.VideoCapture(0)
	while True:
		ret,frame = cap.read()
		
		# 转换hsv
		hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

		# 设定蓝色阈值
		lower_bule = np.array([100,50,50])
		upper_blue = np.array([130,255,255])

		# 根据阈值构建蒙板
		mask = cv.inRange(hsv,lower_bule,upper_blue)
		res = cv.bitwise_and(frame,frame,mask=mask)

		# 显示图像
		cv.imshow('frame',frame)
		cv.imshow('mask',mask)
		cv.imshow('res',res)
		k = cv.waitKey(5) & 0xFF
		if k == 27:
			break
	cv.destroyAllWindows()

def getHSV(colorSpace):
	color = np.uint8([[colorSpace]])
	return cv.cvtColor(color,cv.COLOR_BGR2HSV)[0][0]

if __name__ == '__main__':
	# 提取红黄蓝三种颜色
	getBlue()
	


















