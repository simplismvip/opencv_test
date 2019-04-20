#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 几何变换
# 本小节目标
# 学习对图像进行各种变化：平移，旋转，放射变换
# 本节函数：cv2.getPerspecttivetransForm()

# OpenCV提供了两个变化函数，cv.warpAffine和cv.warpPerspective。
# 使用这两个函数可以实现所有的列项转换，前者接收参数2X3矩阵，后者3X3变换矩阵

# 放缩变化：cv.resize()
# 图像的尺寸可以手动设置，也可以指定缩放因子。选择使用不同的插值方法，缩放时使用cv.INTER_AREA,放大时使用cv.INTER_CUBIC和
# cv.INTER_LINEAR,默认情况下所有改变图片尺寸大小的操作使用的插值方法都是cv.INTER_LINEAR
def change_image_size():
	img = cv.imread('./cv_source/start.png')
	# 下面的None本来是输出图像的尺寸，但是因为后边设置了缩放因子，因而设置为none
	res = cv.resize(img,None,fx = 2,fy = 2,interpolation=cv.INTER_CUBIC)
	cv.imwrite('./cv_source/start_new.png',res)

# 平移
def moveChange():
	img = cv.imread('./cv_source/start.png')
	
	# 构建2X3单位矩阵
	mat = np.eye(2,3)
	# 添加x,y方向移动距离
	mat[0:,2:] = [[50],[100]]
	# 获取形状，构架输出大小
	rows,cols,r = img.shape

	# 平移函数
	res = cv.warpAffine(img,mat,(cols,rows))
	cv.imwrite('./cv_source/start_new.png',res)

# 旋转
# 旋转矩阵百度一下
# 为了构建旋转矩阵，CV提供了一个函数：cv.getRotationMatrix2D()
# 例子提供了一个在不缩放的情况下将图像旋转90度
def remote():
	img = cv.imread('./cv_source/start.png')
	rows,cols,r = img.shape
	# 第一个参数为旋转中心，第二个为旋转角度，第三为旋转后的缩放因子
	# 通过设置旋转中心，旋转因子，以及窗口带下防止旋转后超出边界问题
	mat = cv.getRotationMatrix2D((cols/2,rows/2),45,0.6)
	res = cv.warpAffine(img,mat,(cols,rows))
	cv.imwrite('./cv_source/start_new.png',res)

# 仿射变换
# 仿射变换中，原图中所有的平行线子啊结果图中同样平行
# 为了创建这个矩阵我们需要从原图找到三个点一级他们在输出图像中的位置，
# cv.getAffineTransfrom()会创建一个2x3的矩阵，最后这个矩阵会被传给函数cv.warpAffine()
def fangshebianhuan():
	img = cv.imread('./cv_source/start.png')
	rows,cols,ch = img.shape
	pts1 = np.float32([[50,50],[200,50],[50,200]])
	pts2 = np.float32([[10,100],[200,50],[100,250]])

	mat = cv.getAffineTransform(pts1,pts2)
	print pts1,pts2,mat

	res = cv.warpAffine(img,mat,(cols,rows))
	cv.imwrite('./cv_source/start_new.png',res)
	plt.subplot(121)
	plt.imshow(img,cmap='jet')
	plt.title('Input')

	plt.subplot(122)
	plt.imshow(res,cmap='jet')
	plt.title('Output')
	plt.show()

# 透视变换
# 对于透视变化，我们需要一个3x3的变换矩阵。变换前后直线还是直线。
# 要构建这个变换矩阵，需要在输入图像上找到四个点，以及在输出图像上的对应位置
# 四个点中任意三点不能共线，这个变换矩阵可以使用函数cv.getPerspectiveTransfrom()构建
# 然后把这个函数传给cv.warpPerspective()
def toushibianhua():
	img = cv.imread('./cv_source/toushibianhua.png')
	rows,cols,ch = img.shape
	pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
	pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

	# 两个参数：一个是输出图像上点，一个是输入图像上点
	mat = cv.getPerspectiveTransform(pts1,pts2)
	res = cv.warpPerspective(img,mat,(300,300))
	cv.imwrite('./cv_source/start_new.png',res)

def test():
	cvp = cv.VideoCapture('./cv_source/opencv_testvideo.mp4')
	while True:

		# 获取每一帧
		ret,frame = cap.read()
		hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

		# 设定蓝色阈值，
		lower_blue = np.array([110,50,50])
		upper_blue = np.array([130,255,255])

		mask = cv.inRange(hsv,lower_blue,upper_blue)
		res = cv.bitwise_and(frame,frame,mask=mask)
		cv.imshow('res',res)

		k = cv.waitKey(5) & 0xFF
		if k == 27:
			break
	cv.destroyAllWindows()

if __name__ == '__main__':
	fangshebianhuan()
	




































