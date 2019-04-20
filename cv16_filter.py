#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random

# 这篇文章介绍滤波和平滑：https://blog.csdn.net/xierhacker/article/details/52693775

# 图像平滑
# 本小节目标
# 学习不同低通滤波器对图像进行模糊
# 自定义滤波器对图像进行卷积（2D卷积）

# 2D卷积
# 可以对图像实施低通滤波(LPF)和高通滤波(HPF)。LPF可以帮助去噪声，HPF可以帮助我们找到图像的边缘
# OpenCV提供的cv.filter2D()可以让我们对一幅图像进行卷积操作。
def filter2D():
	img = cv.imread('./cv_source/testimg.jpeg')
	getVoiceImage(img)
	# 构建平均滤波器核
	kernel = np.ones((5,5),np.float32)/25
	# 算法思想：将核放在图像的一个像素A上，求与核对应的图像上25个像素的平均值，
	# 用这个平均值代替像素A的值。重复以上操作指导奖图像的每个像素都更新一遍。
	# 构建滤波核(NxN矩阵)，找到点A和点A的临域(一个NxN矩阵)，计算临域矩阵像素平均值代替点A
	det = cv.filter2D(img,-1,kernel)
	# cv.namedWindow('image')
	# cv.imshow('image',det)
	# cv.waitKey(0)
	# cv.destroyAllWindows()
	pltDraw(121,img,'Origin')
	pltDraw(122,det,'Averaging')
	plt.show()

# 高斯模糊
# 把上述卷积核换成高斯核(即原来矩阵中的数值是相等的，现在换成符合高斯分布。矩阵中心值最大，距离中心元素的距离依次递减，三维空间类似小山包，
# 原来求平均数现在换成求加权平均数，权就是方框里面的值)。CV提供有高斯滤波函数:cv.GaussianBlur()
# 我们只需要指定高斯核的宽和搞，以及高斯函数沿X，Y方向上的标准差，如果只指定X方向或者Y方向，两个方向值相等。如果都指定为0，函数会根据核函数自己计算
# 当然我们可以提供自己的高斯核：cv.getGaussianKernel()可以自己构建一个高斯核
def gaussianBule():
	img = cv.imread('./cv_source/testimg.jpeg')
	getVoiceImage(img)
	det = cv.GaussianBlur(img,(5,5),0)
	cv.namedWindow('image')
	cv.imshow('image',det)
	cv.waitKey(0)
	cv.destroyAllWindows()

# 中值模糊
# 中值滤波常用来去除椒盐噪声
def medianBlur():
	img = cv.imread('./cv_source/testimg.jpeg')
	getVoiceImage(img)
	det = cv.medianBlur(img,5)
	cv.namedWindow('image')
	cv.imshow('image',det)
	cv.waitKey(0)
	cv.destroyAllWindows()

# 双边滤波函数cv.bliaterlFilter()能在保持便捷清晰的情况下有效去除噪声，但是速度较慢
# 我们已经知道，高斯滤波只考虑像素之间的空间关系，而不考虑像素之间的关系（像素的相识度）。因而这种方法不会考虑是否位于边界
# 所以边界也会模糊掉，这不是我们希望的。双边滤波在同事使用空间高斯权重和灰度值相似性高斯权重。空间高斯函数确保只有临近区域像素对中心点有影响，
# 弧度相似性高斯函数确保只有与中心像素灰度值接近才会被用来做模糊运算，所以这种方法会确保边界不会被模糊掉，因为边界处的灰度值变化大。
# 
def bliateralFilter():
	img = cv.imread('./cv_source/testimg.jpeg')
	getVoiceImage(img)
	# 9是临域直径，两个75是空间高斯函数标准差，灰度相似性高斯函数标准差
	det = cv.bliateralFilter(img,97,75,75)
	cv.namedWindow('image')
	cv.imshow('image',det)
	cv.waitKey(0)
	cv.destroyAllWindows()

# 图片随机添加噪点
def getVoiceImage(img):
	# 随机往图片中添加噪点
	w,h,c = img.shape
	wRand = random.sample(range(w-1),205)
	hRand = random.sample(range(w-1),200)
	for w,h in zip(wRand,hRand):
		# 添加白色噪点
		img[w,h] = (255,255,255)
		c = img[w,h]

# 画出图像
def pltDraw(location,img,title):
	plt.subplot(location)
	plt.imshow(img，camp='gray')
	plt.title(title)
	plt.xticks([])
	plt.yticks([])

if __name__ == '__main__':
	gaussianBule()
	




































