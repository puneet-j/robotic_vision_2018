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
mask_use[r[1]:r[1]+r[3],r[0]:r[0]+r[2]] = old_gray[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]
p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask_use, **feature_params)

old_roi = r
######## optical flow
while(1):
    # find dt
    t = time()
    dt = t-t0
    t0 = t

    #read the frame, detect motion
    ret,frame = cap.read()
    mask = np.zeros_like(old_frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # KLT for tracking points in the ROI
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # print("p1",p1,"p0",p0)#,"speed",p1-p0)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        #print i
        #print color[i]
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(frame,(a, b),5,color[i].tolist(),-1)
        if i == 99:
            break

    ###### calculating speed and converting to tuples for showing on image
    speed = good_new - good_old
    x_sp = np.mean(speed[:,0])
    y_sp = np.mean(speed[:,1])
    # print(speed,x_sp,y_sp)
    x_new = good_new[:,0]
    y_new = good_new[:,1]
    x_old = good_old[:,0]
    y_old = good_old[:,1]
    center = (np.mean(x_new),np.mean(y_new))

    # state[0,0] = np.mean(x_old) #r[0] + 0.5*r[2]
    # state[1,0] = np.mean(y_old) #r[1] + 0.5*r[3]

    # add dt to transition matrix for KF
    tm = np.copy(kf.transitionMatrix)
    tm[0,2] = dt
    tm[1,3] = dt
    kf.transitionMatrix = np.copy(tm)
    s = np.copy(state)
    m  = np.copy(meas)
    # predict next position
    # print(s)
    # print(kf.transitionMatrix)
    # print(kf.controlMatrix)
    # print(kf.errorCovPre)
    s = kf.predict()

    m[0] = center[0]
    m[1] = center[1]
    m[2] = x_sp
    m[3] = y_sp
    # print("meas", meas, "cov",np.errorCovPre, "np.")
    # print(s,"state")    
    # print(meas,"meas")
    # print(center,"center",x_sp,y_sp,"speed")
    # print(good_new,good_old)
    if np.all(np.equal(good_new,good_old)):
        continue
    else:
        s = kf.correct(m)
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
    err = kf.errorCovPost
    radius = int(max(r[2],r[3]))
    # print(radius)
    mask_plot_ = cv2.circle(mask, center, radius, (255,0,0),1)
    # mask_plot_ = cv2.rectangle(mask, (int(mask_plot[0]),int(mask_plot[1])),(int(mask_plot[0] + r[2]),int(mask_plot[1] + r[3])),(0,255,255),1)

    img = cv2.add(frame,mask_plot_)
    img = cv2.add(frame,mask)


    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()


