#!/usr/bim/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
import os

# 将一幅图片分割为若干个小图片
def imgAdd(path):
	img = cv.imread(path)
	w = img.shape[0]
	h = img.shape[1]
	rate = min((w//690,h//1000))
	if rate>1:
		print "Rate %s,%s,%s" % (rate,w,h)
		for wRate in range(rate):
			for hRate in range(rate):
				rect = (wRate*960,hRate*1000,690,1000)
				newIma = img[hRate*1000:(hRate+1)*1000,wRate*690:(wRate+1)*690]
				# cv.namedWindow('image')
				# cv.imshow('image',newIma)
				# cv.waitKey(0)
				# cv.destroyAllWindows()

				ps = os.path.split(path)
				newPath = ps[0]+"/"+str(hRate)+str(wRate)+"_"+ps[1]
				cv.imwrite(newPath,newIma)
				print newIma.shape,hRate,wRate,
	else:
		print "too small!"

def splite(path):
	img = cv.imread(path)
	w = img.shape[1]
	h = img.shape[0]
	# 0到h行，0到w/2列
	# 元组中照片行列反了（高，宽）
	ps = os.path.split(path)
	newPath1 = ps[0]+"/"+"1"+ps[1]
	newPath2 = ps[0]+"/"+"2"+ps[1]

	newIma1 = img[0:h,0:w/2]
	cv.imwrite(newPath1,newIma1)

	newIma2 = img[0:h,w/2:w]
	cv.imwrite(newPath2,newIma2)
	print newIma2.shape,(w,h)

if __name__ == '__main__':
	splite('/Users/junming/Desktop/spliteImage/A81818EFqGQ.jpg')
	


















