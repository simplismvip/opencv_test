#!/usr/bim/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv

# 把鼠标当做画笔

# 本小节目标
# 学习OpenCV处理鼠标事件
# 将要学习的函数：cv2.setMouseCallback()

# 在鼠标点击过的位置绘制一个圆圈
# 获取CV支持的鼠标事件
events = [i for i in dir(cv) if 'EVENT' in i]
print events
class MouseEvent(object):
	"""鼠标双击画圆"""
	# event是 CV_EVENT_*变量之一
	# x和y是鼠标指针在图像坐标系的坐标（不是窗口坐标系） 
	# flags是CV_EVENT_FLAG的组合， param是用户定义的传递到setMouseCallback函数调用的参数。	
	def draw_cricle(self,event,x,y,flags,param):
		if event == cv.EVENT_LBUTTONDBLCLK:
			cv.circle(self.img,(x,y),50,(255,255,255),-1)

	def mouseCallback(self):
		self.img = np.zeros((512,512,3),np.uint8)
		cv.namedWindow('image')

		# winname:窗口的名字
		# onMouse:鼠标响应函数，回调函数。指定窗口里每次鼠标时间发生的时候，被调用的函数指针。 
		# 这个函数的原型应该为void on_Mouse(int event, int x, int y, int flags, void* param);  
		cv.setMouseCallback('image',self.draw_cricle)
		while True:
			cv.imshow('image',self.img)
			if cv.waitKey(2) & 0xFF == dir('q'):break
		cv.destroyAllWindows()

class MoveMouseEvent(object):
	def __init__(self):
		super(MoveMouseEvent,self).__init__()
		# 标记鼠标按下，用来切换状态
		self.drawing = False
		# 标记当前画矩形还是圆
		self.mode = True
		self.ix,self.iy = -1,-1

	def draw_circle(self,event,x,y,flags,param):
		if event == cv.EVENT_LBUTTONDOWN:
			self.drawing = True
			self.ix,self.iy = x,y
		elif event == cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
			if self.drawing == True:
				if self.mode == True:
					cv.rectangle(self.img,(self.ix,self.iy),(x,y),(255,255,255),-1)
				else:
					cv.circle(self.img,(x,y),3,(255,255,255),-1)
		elif event == cv.EVENT_LBUTTONUP:
			self.drawing ==False
	
	def startDraw(self):
		self.img = np.zeros((512,512),np.uint8)
		cv.namedWindow('image')
		cv.setMouseCallback('image',self.draw_circle)
		while True:
			cv.imshow('image',self.img)
			k = cv.waitKey(1) & 0xff
			if k == ord('m'):
				self.mode = not self.mode
			elif k == 27:
				break

class MoveMouseDrawUnFill(object):
	"""画未填充矩形"""
	def __init__(self):
		super(MoveMouseDrawUnFill, self).__init__()
		self.ix,self.iy = -1,-1
	
	def draw_rectangle(self,event,x,y,flags,param):
		if event == cv.EVENT_LBUTTONDOWN:
			# 记录初始坐标
			self.ix,self.iy = x,y

		elif event == cv.EVENT_LBUTTONUP:
			# 开始画线
			cv.rectangle(self.img,(self.ix,self.iy),(x,y),(255,255,255),3)

	def startDraw(self):
		self.img = np.zeros((512,512),np.uint8)
		cv.namedWindow('image')
		cv.setMouseCallback('image',self.draw_rectangle)
		while True:
			cv.imshow('image',self.img)
			if cv.waitKey(1) & 0xff == 27:
				break

if __name__ == '__main__':

	# 点击鼠标出画圆
	# mouse = MouseEvent()
	# mouse.mouseCallback()
	
	# 移动鼠标画线
	# move_mouse = MoveMouseEvent()
	# move_mouse.startDraw()

	# 移动鼠标未填充矩形
	m = MoveMouseDrawUnFill()
	m.startDraw()

















