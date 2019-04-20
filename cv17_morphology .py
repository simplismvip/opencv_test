#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random

# 形态学转换
# 本小节目标
# 学习不同不同形态学操作，腐蚀，膨胀，开运算，闭运算等
# 本节函数：cv.erode(),cv.dilate(),cv.morphologyEx()

# 原理：形态学操作时根据图像形状进行的简单操作。一般情况下对二值化图像进行的操作，需要输入两个参数，一个是原始图像，第二个被称为结构化元素或者核，
# 他是用来觉得操作的性质，两个解基本的形态学艹在时腐蚀和膨胀，他们的变体构成了开运算，闭运算，梯度等

# 膨胀就是对图像高亮部分进行“领域扩张”，效果图拥有比原图更大的高亮区域；
# 腐蚀是原图中的高亮区域被蚕食，效果图拥有比原图更小的高亮区域。
# 开运算：先腐蚀再膨胀，用来消除小物体
# 闭运算：先膨胀再腐蚀，用于排除小型黑洞
# 形态学梯度：就是膨胀图与俯视图之差，用于保留物体的边缘轮廓。
# 顶帽：原图像与开运算图之差，用于分离比邻近点亮一些的斑块。
# 黑帽：闭运算与原图像之差，用于分离比邻近点暗一些的斑块。

# 腐蚀
# 这个操作会把前景物体的边界腐蚀掉，基本原理是卷积核沿着图像滑动没如果与卷积核对应的原图的所有像素都是1，那么中心元素就保持原来的像素值，
# 否则变为0.这样的效果是根据卷积核的大小靠近前景的所有像素都会被腐蚀掉（变为0），所以前景物体会变小，政府图像的白色区域会减少，对于去除
# 白噪声很有用，也可以用来断开两个连接在一起的物体
def fuShi():
	img = cv.imread('./cv_source/testimg.jpeg')
	getVoiceImage(img)#img = cv.GaussianBlur(img,(5,5),0)
	kernel = np.ones((5,5),np.uint8)
	l_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	erosion = cv.erode(l_image,kernel,iterations=1)
	cv.namedWindow('img')
	cv.imshow('img',erosion)
	cv.waitKey(0)

def pengZhang():
	img = cv.imread('./cv_source/testimg.jpeg')
	getVoiceImage(img)# cv.GaussianBlur(img,(5,5),0)
	kernel = np.ones((5,5),np.uint8)
	l_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	dilation = cv.dilate(l_image,kernel,iterations=1)
	cv.namedWindow('img')
	cv.imshow('img',dilation)
	cv.waitKey(0)

# 先执行腐蚀去除噪点，在进行膨胀扩展。开运算就用来去除噪声
# cv.morphologyEx()
def kaiYunSuan():
	img = cv.imread('./cv_source/testimg.jpeg')
	getVoiceImage(img)# cv.GaussianBlur(img,(5,5),0)
	kernel = np.ones((5,5),np.uint8)
	l_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	opening = cv.morphologyEx(l_image,cv.MORPH_OPEN,kernel)
	cv.namedWindow('img')
	cv.imshow('img',opening)
	cv.waitKey(0)

# 先膨胀再腐蚀，膨胀去除前景图片黑点，腐蚀还原图像
def biYunSuan():
	img = cv.imread('./cv_source/testimg.jpeg')
	getVoiceImage(img)# cv.GaussianBlur(img,(5,5),0)
	kernel = np.ones((5,5),np.uint8)
	l_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	opening = cv.morphologyEx(l_image,cv.MORPH_CLOSE,kernel)
	cv.namedWindow('img')
	cv.imshow('img',opening)
	cv.waitKey(0)

# 展示一幅图像膨胀和腐蚀的差别，结果看上去就是图像的轮廓
# 就是膨胀和腐蚀之差
def xingTaiXueTiDu():
	img = cv.imread('./cv_source/testimg.jpeg')
	getVoiceImage(img)# cv.GaussianBlur(img,(5,5),0)
	kernel = np.ones((5,5),np.uint8)
	l_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	opening = cv.morphologyEx(l_image,cv.MORPH_GRADIENT,kernel)
	cv.namedWindow('img')
	cv.imshow('img',opening)
	cv.waitKey(0)	

# 原始图像与尽兴开运算之后得倒的图像的差
def liMao():
	img = cv.imread('./cv_source/testimg.jpeg')
	getVoiceImage(img)# cv.GaussianBlur(img,(5,5),0)
	kernel = np.ones((5,5),np.uint8)
	l_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	opening = cv.morphologyEx(l_image,cv.MORPH_TOPHAT,kernel)
	cv.namedWindow('img')
	cv.imshow('img',opening)
	cv.waitKey(0)	

# 原图像与进行闭运算之后的差
def heiMao():
	img = cv.imread('./cv_source/testimg.jpeg')
	getVoiceImage(img)# cv.GaussianBlur(img,(5,5),0)
	kernel = np.ones((5,5),np.uint8)
	l_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	opening = cv.morphologyEx(l_image,cv.MORPH_BLACKHAT,kernel)
	cv.namedWindow('img')
	cv.imshow('img',opening)
	cv.waitKey(0)	

# 结构化元素
# 上面使用的结构化元素是正方形的。有时候需要构建椭圆或圆形的核，这时就需要使用cv提供的函数
# cv.getStructingElement().只需要告诉需要的形状和大小
# https://blog.csdn.net/keen_zuxwang/article/details/72768092
def getStructEmelent():
	# 矩形
	kernel1 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
	# 椭圆
	kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
	# 十字形
	kernel3 = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
	print kernel1
	print kernel2
	print kernel3


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
	plt.imshow(img)
	plt.title(title)
	plt.xticks([])
	plt.yticks([])

if __name__ == '__main__':
	getStructEmelent()
	




































