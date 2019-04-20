#!/usr/bin/python
# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

# 读取一张图片
def showImage(path):
	# 参数：第一个表示路径，第二个表示以何种模式读取照片
	# cv.IMREAD_COLOR：读入彩色图像，图像透明度会被忽略
	# cv.IMREAD_GRAYSCALE：灰度模式读入图像
	# cv.IMREAD_UNCHANGED：读入图像，并且包括图像的alpha通道
	img = cv.imread(path,cv.IMREAD_GRAYSCALE)

	#创建窗口并显示图像
	cv.namedWindow("Image")

	# 显示图像，窗口会自动调整为图像大小，第一个参数为窗口名字，第二个才是图像
	# 可以创建多个窗口，但是必须给予不同的名字
	cv.imshow("Image",img)

	# 是一个􏱠盘绑定函数
	cv.waitKey(0)
	
	# 释放创建的窗口
	# cv.destroyAllWindows()
	cv.destroyWindow('Image')

	# 保存图像，第一个参数：文件名，第二个：要保存的文件
	# 图片显示后。按‘s’键保存后退出，ESC键退出不保存
	cv.imwrite('./newImage.png',img)

# 使用 Matplotlib
from matplotlib import pyplot as plt
def matplot_test(path):
	img = cv.imread(path,0)
	# opencv加载照片使用BGR模式，但是matplotlib使用RGB模式，
	# 所以如果opencv使用im.IMREAD_COLOR模式加载彩色照片，matplotlib读取时将不会正确显示
	plt.imshow(img,cmap='gray',interpolation='bicubic')
	plt.xticks([])
	plt.yticks([])
	plt.show()

if __name__ == '__main__':
	# showImage('./start.png')
	matplot_test('./start.png')































