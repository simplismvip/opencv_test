#!/usr/bim/python
# -*- coding:utf-8 -*-

import cv2,time

# 本小节目标
# 1、学会读取，显示、保存视频文件
# 2、学会从摄像头获取并显示视频
# 3、学会使用一下函数：cv2.VideoCapture(),cv2.VideoWrite()

# 用摄像头捕获视频
# VideoCapture对象用来获取视频。参数可以是视频索引，或者是一个视频文件。设备索引号指的是要使用的摄像头，
# 笔记本一般有内置摄像头，可以设置0获取，也可以设置1或者其他选择摄像头
def getVideoFromCapture():
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		cap.open(0)

	# cap.get(proId)用来获取视频参数信息，propid范围0-18，每个数代表一个属性。具体含义参考一下：
	# CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
		# CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
	# CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of the film, 1 - end of the film.
	# CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
	# CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
	# CV_CAP_PROP_FPS Frame rate.
	# CV_CAP_PROP_FOURCC 4-character code of codec.
	# CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
	# CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
	# CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
	# CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
	# CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
	# CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
	# CV_CAP_PROP_HUE Hue of the image (only for cameras).
	# CV_CAP_PROP_GAIN Gain of the image (only for cameras).
	# CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
	# CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
	# CV_CAP_PROP_WHITE_BALANCE Currently unsupported
	# CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend cur- rently)
	# width = cap.get(3)
	# heigth = cap.get(4)
	# print cap.get(5)
	
	# cap.set(propid,value):用来修改对应的值
	# 例如：修改视频宽高：cap.set(3,320),cap.set(4,240)
	cap.set(3,640),cap.set(4,480)

	while (cap.isOpened()):
		# cap.read()：返回布尔值。
		ret,frame = cap.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		cv2.imshow('frame',gray)

		# 监听键盘输入，如果输入‘q’,退出程序
		if cv2.waitKey(1) & 0xff == ord('q'):break
		if cv2.waitKey(1) == ord('s'):
			imageName = '%.0d' % time.time()+'.png'
			cv2.imwrite(imageName,gray)

	cap.release()
	cv2.destroyAllWindows()

# 读取视频只需要把cv2.VideoCapture()函数参数换为路径即可
# cv2.waitKey(25)函数的意义是等待一个输入，参数为0表示永久等待。参考下面链接
# https://blog.csdn.net/u014737138/article/details/80375514
def getVideoFromFile(path):
	cap = cv2.VideoCapture(path)
	while (cap.isOpened()):
		ret,frame = cap.read()
		grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		cv2.imshow('vodeo',grayFrame)
		if cv2.waitKey(25) & 0xff == ord('q'):break
	cap.release()
	cv2.destroyAllWindows()

def saveVideo(savePath):
	cap = cv2.VideoCapture(0)
	cap.set(3,480)
	cap.set(4,640)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	out = cv2.VideoWriter(savePath,fourcc,20,(640,480))
	while (cap.isOpened()):
		ret,frame = cap.read()
		if ret == True:
			# cv2.flip():用来翻转图像。参数：1：水平翻转，0垂直翻转，-1：水平垂直翻转
			# frame = cv2.flip(frame,0)
			# frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			out.write(frame)
			cv2.imshow('frame',frame)
			if cv2.waitKey(1) & 0xff == ord('q'):break
		else:
			break
	cap.release()
	out.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	# 获取视频
	getVideoFromCapture()
	# getVideoFromFile('./opencv_testvideo.mp4')
	# saveVideo('./output.avi')



















