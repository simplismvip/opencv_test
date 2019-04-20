#!/usr/bim/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
############ 第三部分：核心操作 ############

# 图像的基础操作：本小节目标
# 获取像素值并且修改
# 获取图像的属性信息
# 图像的ROI()
# 图像通道的拆分和合并
# 照片有像素点组成：长 X 宽，每个像素点有(B,G,R)组成。三种组合构成一个(长，宽，3)的矩阵
# 因此一张690X459的照片可以表示为(690, 459, 3)的矩阵
# 如果是灰度图，第三个维度的不是3

def test1():
	img = cv.imread('../cv_source/testimg.jpeg')
	# 获取矩阵维度
	# print img.shape

	# 获取(100,100)这个点的像素值
	px = img[100,100]
	print px

	# 获取(100,100)这个像素点的B值
	B = img[100,100,0]
	print B

	# 修改(100,100)像素值
	img[100,100] = [255,255,255]

	print img[100,100]

	# 使用切片获取一个区域的数值
	# print img[]
	
def test2():
	img = cv.imread('../cv_source/testimg.jpeg')
	print img.item(10,10,2)
	img.itemset((10,10,2),100)
	print img.item(10,10,2)

# 获取图像的属性：包括行，列，通道，图像数据类型，像素数
# 这些属性都是numpy的属性
def imgProperty():
	img = cv.imread('../cv_source/testimg.jpeg')

	# 获取图像的形状
	print img.shape

	# 获取图像像素数
	print img.size

	# 图像数据类型
	print img.dtype

# 需要对图像特定区域操作
def imgROI():
	img = cv.imread('../cv_source/testimg.jpeg')	
	boll = img[60:160,356:456]
	img[280:380,260:360] = boll
	cv.namedWindow('image')
	cv.imshow('image',img)
	cv.waitKey(0)

# 拆分以及合并图像通道
def getRGB():
	img = cv.imread('../cv_source/testimg.jpeg')
	b,g,r = cv.split(img)
	img = cv.merge([b,g,r])
	# BGR
	# imaR[:,:,:2] = 0
	cv.imshow('image',img)
	cv.namedWindow('image')
	cv.waitKey(0)

# 为图像扩充
def addborder():
	# 扩充图像可以使用cv.coptMakeBorder()
	# top,bottom,left,right对应边界数目
	# bordrttype：要添加的边界类型
	# cv2.BORDER_CONSTANT 要添加有􏰿色的常数值􏰩边界，􏰈􏰒􏰄􏰅下一个参数􏰇value􏰉。
	# cv2.BORDER_REFLECT 边界元素镜像，􏰈􏰒􏰄􏰅下一个参数􏰇value􏰉。
	# cv2.BORDER_REFLECT_101 
	# cv2.BORDER_REPLICATE 重复最后一个元素
	# cv2.BORDER_WRAP
	blue = [255,0,0]
	img1 = cv.imread('../cv_source/testimg.jpeg')

if __name__ == '__main__':
	addborder()

	


















