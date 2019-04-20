#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random

# 边缘检测
# 本小节目标
# 了解CV中的Canny边缘检测
# 本节函数：cv.Canny()

# 原理：Canny边缘检测是一种很流行的边缘检测算法，是john在1986年提出。边缘检测很容易受到噪声影响，所以第一步是使用5x5高斯滤波器去除噪声
# 第二步对平滑后的图像使用Sobel算子计算水平和竖直方向的一阶导数倒数，根据得到的两幅梯度图找到边界的梯度和方向


# CV中的Canny边界检测
# 在CV中只需要一个函数:cv.Canny()，
# 参数解释：1、第一个参数是输入图像，灰度图。
# 2、第二第三是minVal和maxValue。其中小阈值用来控制边缘连接，大阈值用来控制强边缘的初始分割，即如果一个像素的梯度大于上限值，认为是边缘像素，小于下线阈值
# 则会被抛弃，如果在两者之间则这个点与高于上限值的像素连接时才会被保留，否则删除。
# 3、第四个参数Sobel算子大小，用来计算
# 梯度图像的Sobel卷积核大小，默认是3，表示3X3矩阵。最后一个参数是L2gradient,他可以设定求梯度大小的方程，如果设为Ture,就会适应上面适应的方程，默认值为false
def canny():
	img = cv.imread("./cv_source/adaptiveThreshold.png",0)
	edges = cv.Canny(img,100,200)
	pltDraw(121,img,"origin")
	pltDraw(122,edges,"Canny")
	plt.show()

def cannyMin(values):
	print values

def creatTrackBar():
	img = cv.imread("./cv_source/testimg.jpeg",0)
	cv.namedWindow('image')
	cv.createTrackbar("maxValue","image",0,255,cannyMin)
	cv.createTrackbar("minValue","image",0,255,cannyMin)
	while True:
		maxValue = cv.getTrackbarPos("maxValue","image")
		minValue = cv.getTrackbarPos("minValue","image")
		edges = cv.Canny(img,minValue,maxValue)
		
		cv.imshow('image',edges)
		k = cv.waitKey(1) & 0xff
		if k == 27:break
	cv.destroyAllWindows()

# 读取视频进行边缘检测
def readVideo():
	cap = cv.VideoCapture(0)
	cap.set(3,480)
	cap.set(4,640)
	
	cv.namedWindow('image')
	cv.createTrackbar("maxValue","image",0,255,cannyMin)
	cv.createTrackbar("minValue","image",0,255,cannyMin)
	while cap.isOpened():
		ret,frame = cap.read()
		# 生成灰度图
		grayImg = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
		maxValue = cv.getTrackbarPos("maxValue","image")
		minValue = cv.getTrackbarPos("minValue","image")

		edges = cv.Canny(grayImg,minValue,maxValue)
		cv.imshow('image',edges)

		# 监听键盘输入，如果输入‘q’,退出程序
		if cv.waitKey(1) & 0xff == ord('q'):break
		if cv.waitKey(1) == ord('s'):
			imageName = '%.0d' % time.time()+'.png'
			cv.imwrite(imageName,edges)
	cap.release()
	cv.destroyAllWindows()

# 图片随机添加噪点
def getVoiceImage(img):
	# 随机往图片中添加噪点
	w,h,c = img.shape
	wRand = random.sample(range(w-1),205)
	hRand = random.sample(range(w-1),200)
	for w,h in zip(wRand,hRand):
		# 添加白色噪点
		img[w,h] = (0,0,0)
		c = img[w,h]

# 画出图像
def pltDraw(location,img,title):
	plt.subplot(location)
	plt.imshow(img,cmap='gray')
	plt.title(title)
	plt.xticks([])
	plt.yticks([])
if __name__ == '__main__':
	readVideo()
	




































