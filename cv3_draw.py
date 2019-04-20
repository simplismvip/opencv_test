#!/usr/bim/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv

# 本小节目标
# 1、学会使用OpenCV绘制不同几何图形
# 2、学会cv2.line(),cv2.circle(),cv2.rectangle(),cv2.elipse(),cv2.putText()
# 参数：上面所有绘图函数需要设置下面参数
# img:想要绘制图像的那副照片
# color: 形状的颜色。以RGB为例，需要输入一个元组，例如：(255,0,0)代表蓝色
# thickness:线条粗细。如果给一个闭合图形设置为-1，则被填充。默认是1
# linetype:线条类型。8连接，抗锯齿等。默认是8连接，cv2.LINE_AA为抗锯齿
def draw_line():
	img = np.zeros((512,512,3),np.uint8)
	
	# 画直线
	cv.line(img,(0,0),(511,511),(255,0,0),5)

	# 画矩形
	cv.rectangle(img,(100,100),(300,300),(255,255,255),3)

	# 画圆
	cv.circle(img,(256,256),100,(255,255,255),1)

	# 画椭圆。参数含义：中心点，（长半轴，短半轴长度），椭圆沿时针方向旋转的角度，起始角度，结束角度
	cv.ellipse(img,(256,256),(100,250),45,0,360,(255,255,255),3)
	# 画多边形
	pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
	pts = pts.reshape((-1,1,2))
	cv.polylines(img,pts,1,(255,255,255),5)
	cv.polylines
	# 画文字。需要设置参数如下
	# 要绘制的文字
	# 要绘制的位置
	# 字体类型，字体大小，字体的属性（颜色。粗细，线条等）
	font  = cv.FONT_HERSHEY_SIMPLEX
	cv.putText(img,'This a OpenCV test font!',(70,30),font,1,(255,255,255),3)

	winname = 'drawline'
	cv.namedWindow(winname)
	cv.imshow(winname,img)
	cv.waitKey(0)
	cv.destroyAllWindows()
if __name__ == '__main__':
	draw_line()



















