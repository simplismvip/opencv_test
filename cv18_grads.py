#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random

# 图像梯度
# 本小节目标
# 图像梯度，图像边界等
# 本节函数：cv.Sobel(),cv.Schar(),cv.Laplachian()等

# 原理：梯度简单说来就是求导，CV提供三种不同的梯度滤波，Sobel,Scharr,Laplacian
# Sobel,Scharr其实就是求一阶或者二阶导数。Scharr是对Sobel的优化。Laplacian是求二阶导数。


# Sobel算子和Scharr算子
# Sobel算子是高斯平滑与微分操作的结合体，所以它的抗噪声能力很好，你可以设定求导的方向(xorder或者yorder)。还可以设定使用的卷积核(ksize)的大小，
# 如果ksize=-1，会使用3X3的Scharr滤波器，它的效果要比3X3的Sobel滤波器好，所以子啊使用3X3滤波器时应该尽量使用Scharr滤波器，3X3的Scharr滤波器卷积核如下：
# x方向[[-3,0,3],[-10,0,10],[-3,0,3]],y方向[[-3,-10,-3],[0,0,0],[3,10,3]]

# Laplacian算子
# 拉普拉斯算子可以使用二阶导数的形式定义，可以设其理算实现类似于二阶Sobel导数，事实上，CV在计算拉普拉斯算子时直接调用Sobel算子
# 拉普拉斯滤波器使用的卷积核[[0,1,0],[1,-4,1],[0,1,0]]
def laplacian():
	img = cv.imread("./cv_source/testimg.jpeg",0)
	
	# 使用高斯滤波去除噪点
	img = cv.GaussianBlur(img,(5,5),0)

	laplacian = cv.Laplacian(img,cv.CV_64F)
	sobelX = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
	sobelY = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

	pltDraw(221,img,'origin')
	pltDraw(222,laplacian,'laplacian')
	pltDraw(223,sobelX,'sobelX')
	pltDraw(224,sobelY,'sobelY')
	plt.show()
	

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
	laplacian()
	




































