import numpy as np
import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)
import cv2
from time import time


cap = cv2.VideoCapture('mv2_001.avi')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 2,
                       blockSize = 3 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
a = old_frame.shape



r = cv2.selectROI('frame',old_frame, False)
# print(a)
# print(r)
kf = cv2.KalmanFilter(4,4,0,cv2.CV_32F)
kf.measurementMatrix = np.eye(4,dtype=np.float32)
# kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]],np.float32)
kf.processNoiseCov = 1e-5 * np.eye(4,dtype=np.float32)
kf.measurementNoiseCov = 1e-2 * np.eye(4,dtype=np.float32)   
kf.errorCovPost = np.eye(4,dtype=np.float32)

state = np.zeros((4,1),dtype=np.float32)   
kf.errorCovPre = np.eye(4,dtype=np.float32)
state[0,0] = r[0] + 0.5*r[2]
state[1,0] = r[1] + 0.5*r[3]
kf.statePost = state
# kf.statePost = np.random.randn(4, 1) + state
meas = np.zeros((4,1), dtype=np.float32)
t0 = time()
mask_use = np.zeros_like(old_gray)

frame_sub = abs(old_gray) #abs(frame_gray) 
# ret,thresh1 = cv2.threshold(frame_sub,100,255,cv2.THRESH_BINARY)
k1 = np.ones((3,3),np.uint8)
kernel = np.ones((1,1),np.uint8)
erosion = cv2.erode(frame_sub,kernel,iterations = 2)
dilated = cv2.dilate(erosion,k1,iterations = 1) 
e = cv2.erode(dilated,k1,iterations = 1)
# old_gray = e.copy()
mask_use[r[1]:r[1]+r[3],r[0]:r[0]+r[2]] = e[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]
p0 = cv2.goodFeaturesToTrack(e, mask = mask_use, **feature_params)
# print(p0)
old_roi = r
e1 = np.copy(e)
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# # Change thresholds
# params.minThreshold = 10;
# params.maxThreshold = 200;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 100
x_old = int(r[1] + r[3]/2)
y_old = int(r[0] + r[2]/2)
######## optical flow
while(1):
    # find dt
    t = time()
    dt = t-t0
    t0 = t
    # add dt to transition matrix for KF
    tm = np.copy(kf.transitionMatrix)
    tm[0,2] = dt
    tm[1,3] = dt
    kf.transitionMatrix = np.copy(tm)
    s = np.copy(state)
    m  = np.copy(meas)
    # predict next position
    s = kf.predict()




    #read the frame, detect motion
    ret,frame = cap.read()
    mask = np.zeros_like(old_frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # hue ,saturation ,value = cv2.split(frame_gray)
    # KLT for tracking points in the ROI
    # print("p1",p1,"p0",p0)#,"speed",p1-p0)
    frame_sub = abs(frame_gray - old_gray)
    ret,thresh1 = cv2.threshold(frame_sub,100,255,cv2.THRESH_BINARY)
    thresh1 = cv2.medianBlur(thresh1,7)

    # # k1 = np.ones((5,5),np.uint8)
    # kernel = np.ones((5,5),np.uint8)
    kernel2 = np.ones((3,3),np.uint8)
    # dilated = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel2,iterations=2)
    # dilated = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel,iterations=2)

    erosion = cv2.erode(thresh1,kernel,iterations = 1)
    dilated = cv2.dilate(erosion,k1,iterations = 3) 
    # dilated = cv2.dilate(dilated,kernel,iterations = 1)
    # e2 = cv2.erode(dilated,kernel,iterations = 1)
    e2 = dilated.copy()
    # e2 = frame_sub.copy()
    cnts = cv2.findContours(e2, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


    # erosion = cv2.erode(thresh1,kernel,iterations = 2)
    # dilated = cv2.dilate(e,kernel,iterations = 5) 
    # cv2.waitKey(5000)
    # exit()
    # Select good points
    # frame_gray = e2.copy()
    # p1, st, err = cv2.calcOpticalFlowPyrLK(e1, e2, p0, None, **lk_params)
    # Set up the detector with default parameters.
    # detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    # keypoints = detector.detect(dilated)
    # keypoints = detector.detect(e2)
    # im_with_keypoints = cv2.drawKeypoints(e2, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("frame2",e2)


    # try:
    #     good_new = p1[st==1]
    #     good_old = p0[st==1]

    #     # draw the tracks
    #     # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     #     #print i
    #     #     #print color[i]
    #     #     a,b = new.ravel()
    #     #     c,d = old.ravel()
    #     #     cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     #     cv2.circle(frame,(a, b),5,color[i].tolist(),-1)
    #     #     if i == 99:
    #     #         break
    # ###### calculating speed and converting to tuples for showing on image
    print(center)
    if len(cnts) == 0:
        print("no update")
    else:
        # speed = good_new - good_old
        # print(speed)
        x_sp = x-x_old#np.mean(speed[:,0])
        y_sp = y-y_old#np.mean(speed[:,1])
        # print(speed,x_sp,y_sp)
        # x_new = x#good_new[:,0]
        # y_new = y#good_new[:,1]
        x_old = x#good_old[:,0]
        y_old = y#good_old[:,1]
        # center = (np.mean(x_new),np.mean(y_new))
        m[0] = center[0]
        m[1] = center[1]
        m[2] = x_sp
        m[3] = y_sp
        s = kf.correct(m)
    # except:
    #     print("no points")

    # print(s)
    # print(m)
    state  = np.copy(s)
    meas = np.copy(m)
    # kf.update
    # p1_ = p1[:,0,:]
    # p1_ = tuple(map(tuple,p1_))
    # p0_ = p0[:,0,:]
    # p0_ = tuple(map(tuple,p0_))
    # mask_plot = (r[0],r[1])
    center = (s[0],s[1])
    err = kf.errorCovPost
    radius = int(max(r[2],r[3]))
    # print(radius)
    mask_plot_ = cv2.circle(mask, center, radius, (255,0,0),1)
    # mask_plot_ = cv2.rectangle(mask, (int(mask_plot[0]),int(mask_plot[1])),(int(mask_plot[0] + r[2]),int(mask_plot[1] + r[3])),(0,255,255),1)

    img = cv2.add(frame,mask_plot_)
    img = cv2.add(frame,mask)

    # im_with_keypoints = cv2.drawKeypoints(e, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    e1 = e2.copy()
    # p0 = good_new.reshape(-1,1,2)
    # print(e2.shape())
    # p0_new = cv2.goodFeaturesToTrack(old_gray, mask = e2[int(center-r[3]/2):int(center+r[3]/2),int(center-r[2]/2):int(center+r[2]/2)], **feature_params)
    # if p0_new is not None:
    #     p0 = p0_new
cv2.destroyAllWindows()
cap.release()


