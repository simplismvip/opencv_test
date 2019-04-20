#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 图像阈值
# 本小节目标
# 本节将学习简单阈值，自适应阈值，Ots's二值化等
# 本节函数：cv.threshold(),cv.adaptiveThreshold()

# 1、简单阈值
# 与名字一样，这种方法很简单，当像素值高于阈值时，我们给这个像素一个新值，否则另一种颜色
# 这个函数就是cv.threshould()	：参数：1、原图像（应该是灰度图）2、阈值。3、像素高于阈值时被赋予的新的像素值。4、flag
# 解释：https://blog.csdn.net/u011430438/article/details/50583335
# 阈值类型：dsti：输出像素，srci：输入图像像素，T：阈值，M：最大值（max_value）
# CV_THRESH_BINARY，      dsti=(srci > T)?M:0
# V_THRESH_BINARY_INV，   dsti=(srci>T)?0:M
# CV_THRESH_TRUNC    ，   dsti=(srci>T)?M:srci
# CV_THRESH_TOZERO_INV，  dsti=(srci>T)?0:srci
# CV_THRESH_TOZERO，      dsti=(srci>T)?srci:0  
# 函数返回值：1、retVal。2、阈值化之后的输出结果图像
def threshouldImage():
	img = cv.imread('./cv_source/originimage.png')
	gray_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	ret,des1 = cv.threshold(gray_image,127,255,cv.THRESH_BINARY)
	ret,des2 = cv.threshold(gray_image,127,255,cv.THRESH_BINARY_INV)
	ret,des3 = cv.threshold(gray_image,127,255,cv.THRESH_TRUNC)
	ret,des4 = cv.threshold(gray_image,127,255,cv.THRESH_TOZERO)
	ret,des5 = cv.threshold(gray_image,127,255,cv.THRESH_TOZERO_INV)
	titles = ['Origin Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
	images = [img,des1,des2,des3,des4,des5]
	for i in xrange(6):
		index = i+1
		plt.subplot(2,3,index)
		plt.imshow(images[i],'gray')
		plt.title(titles[i])
		plt.xticks([])
		plt.yticks([])
	plt.show()

# 自适应阈值
# 上面我们是采取全局阈值，整幅图像采取一个数作为阈值，这种方法不是适合所有状况，尤其是图片不同部分具有不同亮度的时候，
# 这时需要采取自适应阈值，此时阈值是根据图像上的每一个小区域计算语气对应的阈值，因此在同一幅图像上的不同区域采用的是不同阈值
# 从而使我们能够在亮度不同的情况下去个更好的结果。
# cv.adaptiveThreshold()参数：1、输入图像，2、输出像素最大值，3、指定阈值计算方大，4、临阈大小。5、常数，阈值等于的平均值或者加权平均值减去这个常数
# 阈值计算方法选项有两个：1、ADAPTIVE_THRESH_MEAN_C的计算方法是计算出领域的平均值再减去第七个参数double C的值
# 2、ADAPTIVE_THRESH_GAUSSIAN_C的计算方法是计算出领域的高斯均值再减去第七个参数double C的值
def autoThreshould():
	cv.imread
	img = cv.imread('./cv_source/adaptiveThreshold.png',0)
	# 中值滤波
	img = cv.medianBlur(img,5)
	ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
	
	gray_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	# 参数注意：输入图一定是灰度图，Block Size（临域大熊是个奇数）
	th2 = cv.adaptiveThreshold(gray_image,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
	th3 = cv.adaptiveThreshold(gray_image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

	titles = ['Original Image', 'Global Thresholding (v = 127)','Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
	images = [img,th1,th2,th2]
	for i in xrange(4):	
		plt.subplot(2,2,i+1)
		plt.imshow(images[i],'gray')
		plt.title(titles[i])
		plt.xticks([])
		plt.yticks([])
	plt.show()

# Otsu二值化
# 第一部分提到过retval，在使用Otsu's的时候会用到。
# 在使用全局阈值时，我们随便给出一个数值，我们如何判断选取的数的好坏呢？只有不停试，如果是一幅双峰图
# 我们可以在峰谷之间选一个值作为阈值，这就是Otsu二值化。简单俩说就是对一幅双峰图图像自动根据其直方图计算出一个阈值
# 注意：对于非双峰图，这种方法得到的结果可能不理想
# 这里要用的函数还是cv.threshold(),但是需要多传入一个参数：flag：cv.THRESH_OTUS。这时要把阈值设为0，
# 然后算法会找到最优阈值，这个最优质就是返回值retVal。如果不使用Otsu二值优化，返回的retVal值与设定的阈值相等
def OtsuThreshold():
	img = cv.imread('./cv_source/voiceimage.jpg',0)

	# 全局阈值
	ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

	# Otsu二值化
	ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

	# 先使用5x5的高斯核去噪音，再使用Otsu二值化
	# (5,5)为高斯核的大小，0为标准差
	blur = cv.GaussianBlur(img,(5,5),0)
	ret3,th3 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

	images = [img,0,th1,img,0,th2,blur,0,th3]
	titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)','Original Noisy Image','Histogram',"Otsu's Thresholding",'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

	# plt.subplot()函数使用方法：https://blog.csdn.net/tuxinbang/article/details/76039496
	for i in xrange(3):
		plt.subplot(3,3,i*3+1)
		plt.imshow(images[i*3],'gray')
		plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
		plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
		plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
		plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
		plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
	plt.show()

if __name__ == '__main__':
	OtsuThreshold()
	




































