#!/usr/bim/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv

# 图像上的算术运算：本小节目标
# 学习图像上的算术运算,加法，减法，位运算
# 本节函数：cv2.add(),cv.addWeighted()

# 图像加法
# 使用函数cv.add()将两幅图进行加氟运算，也可以直接使用np,res = img1+img2
# OpenCV中加法是饱和运算：超过255会按照255处理。np中加法是模运算
def imgAdd():
	img1 = cv.imread('./cv_source/test1.png')
	img2 = cv.imread('./cv_source/testimg.jpeg')
	# x = np.uint8([250])
	# y = np.uint8([10])
	
	# cv相加
	img3 = cv.add(img1,img2)
	
	# 直接相加
	# img3 = img1+img2
	# cv.imwrite('../cv_source/test2.png',img3)
	cv.namedWindow('image')
	cv.imshow('image',img3)
	cv.waitKey(0)
	cv.destroyAllWindows()

# 图像混合
# 混合其实是加权相加，公式如下：g(x) = (1 − α) f0(x) + αf1(x)
# 通过修改a的值(0->1)实现非常酷的混合
def imgMix():
	img1 = cv.imread('./cv_source/test1.png')
	img2 = cv.imread('./cv_source/testimg.jpeg')
	
	# 采用公式:dst = α · img1 + β · img2 + γ。这里γ取值0
	dst = cv.addWeighted(img1,0.7,img2,0.3,0)
	cv.imshow('dst',dst)
	cv.waitKey(0)
	cv.destroyAllWindows()

# 按位运算
# 位运算包括：AND,OR,NOT,OXR等
def imgbyte():
	
	# 前景
	img2 = cv.imread('./cv_source/opencv.jpeg')
	# 背景
	img1 = cv.imread('./cv_source/testimg.jpeg')
	rows,cols,channels = img2.shape
	
	# 这一步使用roi在背景图中挖出和前景图大小一样的图片
	roi = img1[0:rows,0:cols]

	# 前景图转灰度
	img2gary = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
	# threshold的用法，第一位为原图（只能是单通道图一般为灰度图），
	# 第二个是阈值，大于阈值的像素点取第三位的数字，最后一个是二值化操作类型
	ret,mask = cv.threshold(img2gary,175,255,cv.THRESH_BINARY)
	# 对mask取反(黑变白，白变黑)
	mask_inv = cv.bitwise_not(mask)

	# 自己与自己AND运算，mask的作用在于前面两幅图AND后再与掩码做AND，
	# 使原图中掩码为1的像素变为1（全黑）
	img1_bg = cv.bitwise_and(roi,roi,mask = mask)
	img2_bg = cv.bitwise_and(img2,img2,mask = mask_inv)

	# 混合logo和取出的部分
	dst = cv.add(img1_bg,img2_bg)
	
	# 放回背景原处
	img1[0:rows,0:cols] = dst
	
	cv.imshow('res',img1)
	cv.waitKey(0)
	cv.destroyAllWindows()

# 抠图主要原理
# 
def test_imagbyte():
	ima_fg = cv.imread('./cv_source/fg.jpg')
	ima_bg = cv.imread('./cv_source/bg.jpeg')

	# 从背景图中挖出前景图大小
	rows,colors,channels = ima_fg.shape
	roi = ima_bg[0:rows,0:colors]

	# 对前景图片转灰度
	ima_fg_gray = cv.cvtColor(ima_fg,cv.COLOR_BGR2GRAY)

	# 根据阈值，将像素值大于200的都转换为255（显示为白色），小于200的转换为0(THRESH_BINARY这种模式下)
	# https://blog.csdn.net/iracer/article/details/49232703
	# 这样的话，logo就转换为黑色(像素0)，其他白色(255)
	ret,dst = cv.threshold(ima_fg_gray,200,255,cv.THRESH_BINARY)

	# 否运算
	mask = cv.bitwise_not(dst)

	mask_fg = cv.bitwise_and(ima_fg,ima_fg,mask=mask)

	mask_bg = cv.bitwise_and(roi,roi,mask=dst)

	ima_bg[0:rows,0:colors] = cv.add(mask_fg,mask_bg)

	cv.imwrite('./cv_source/new_bg.png',ima_bg)
	cv.imshow('mask',ima_bg)
	cv.waitKey(0)
	cv.destroyAllWindows()


# 这部分可以结合前两部分的代码做一个画板应用
if __name__ == '__main__':
	# imgAdd()
	# imgMix()
	test_imagbyte()
	


















