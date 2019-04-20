#!/usr/bim/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv

# 用滑动条做调色板

# 本小节目标
# 学习把滑动条绑定搭配OpenCV窗口
# 将要学习的函数：cv2.getTranchbarPos(),cv2.creatTrackbar()
# 创建程序：要求通过调节滑动条设定画板颜色，创建一个窗口显示颜色，三个滑动条设置BGR颜色
# cv2.getTranchbarPos()函数第一个参数是滑动条名字，第二个是被放置的窗口的名字，第三是默认位置，第四个是默认值，第五个是回调函数
def nothing(x):
	print x

img = np.zeros((300,512,3),np.uint8)
cv.namedWindow('image')
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)

switch = '0:OFF\n1:ON'
cv.createTrackbar(switch,'image',0,1,nothing)
while True:
	cv.imshow('image',img)
	k = cv.waitKey(1) & 0xff
	if k == 27:break
	r = cv.getTrackbarPos('R','image')
	g = cv.getTrackbarPos('G','image')
	b = cv.getTrackbarPos('G','image')
	s = cv.getTrackbarPos(switch,'image')
	print r,g,b
	if s == 0:
		img[:] = 0
	else:
		img[:]=[r,g,b]
cv.destroyAllWindows()

# 这部分可以结合前两部分的代码做一个画板应用
if __name__ == '__main__':
	pass

	


