import numpy as np
import cv2
# import thread, time


vid = cv2.VideoCapture(0)
ret,im = vid.read()
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
cv2.imshow('video',im)
cv2.imshow('video_gray',im_gray)

while(True):

	if cv2.waitKey(10) & 0xFF == ord('n'):
		while(True):
			ret,im = vid.read()
			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
			print("hi I have returned to normal")
			cv2.imshow('video',im)
			cv2.imshow('video_gray',im_gray)
			if cv2.waitKey(10) & 0xFF == ord('o'):
				break

	elif cv2.waitKey(10) & 0xFF == ord('b'):
		while(True):
			ret,im = vid.read()
			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
			im = cv2.blur(im,(7,10))
			im_gray = cv2.blur(im_gray,(7,10))
			print ("hi i am in blur")
			cv2.imshow('video',im)
			cv2.imshow('video_gray',im_gray)
			if cv2.waitKey(10) & 0xFF == ord('o'):
				break

	elif cv2.waitKey(10) & 0xFF == ord('e'):
		while(True):
			ret,im = vid.read()
			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
			im = cv2.Canny(im,100,200)
			im_gray = cv2.Canny(im_gray,100,200)
			print("hi i am in canny")
			cv2.imshow('video',im)
			cv2.imshow('video_gray',im_gray)
			if cv2.waitKey(10) & 0xFF == ord('o'):
				break

	elif cv2.waitKey(10) & 0xFF == ord('v'):
		while(True):
			ret,im = vid.read()
			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
			im = cv2.Sobel(im,-1, 1,0,3,1,10)
			im_gray = cv2.Sobel(im_gray,-1, 1,0,3,1,10)
			print("hi i am in sobel vertical")
			cv2.imshow('video',im)
			cv2.imshow('video_gray',im_gray)
			if cv2.waitKey(10) & 0xFF == ord('o'):
				break

	elif cv2.waitKey(10) & 0xFF == ord('h'):
		while(True):
			ret,im = vid.read()
			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
			im = cv2.Sobel(im,-1, 0,1,3,1,10)
			im_gray = cv2.Sobel(im_gray,-1, 0,1,3,1,10)
			print("hi i am in sobel horizontal")
			cv2.imshow('video',im)
			cv2.imshow('video_gray',im_gray)
			if cv2.waitKey(10) & 0xFF == ord('o'):
				break

	elif cv2.waitKey(10) & 0xFF == ord('z'):
		while(True):
			ret,im = vid.read()
			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
			im = cv2.filter2D(im,-1,np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]],np.float32))#,(-1,-1),10)
			im_gray = cv2.filter2D(im_gray,-1,np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))#,(-1,-1),10)
			print("hi i am in custom sharpening")
			cv2.imshow('video',im)
			cv2.imshow('video_gray',im_gray)
			if cv2.waitKey(10) & 0xFF == ord('o'):
				break


	else:
		ret,im = vid.read()
		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
		# print "hi I have returned to normal"
		cv2.imshow('video',im)
		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
		cv2.imshow('video_gray',im_gray)
		print ("hi I have returned to normal")

	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

vid.release()
cv2.destroyAllWindows()
