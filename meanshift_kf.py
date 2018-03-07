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

'''

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv.imshow('img2',img2)
        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k)+".jpg",img2)
    else:
        break
'''

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
# mask_use = np.zeros_like(old_gray)
# mask_use[r[1]:r[1]+r[3],r[0]:r[0]+r[2]] = old_gray[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask_use, **feature_params)
roi = old_frame[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1 )
# old_rect = np.copy(r)

# print(r)
old_roi = r
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
    print(s,"before predict")
    # predict next position
    # print(s)
    # print(kf.transitionMatrix)
    # print(kf.controlMatrix)
    # print(kf.errorCovPre)
    s = kf.predict()

    print(s,'after predict')
    #read the frame, detect motion
    ret,frame = cap.read()
    mask = np.zeros_like(old_frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # roi = frame[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]
    # hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    # KLT for tracking points in the ROI
    # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # print("p1",p1,"p0",p0)#,"speed",p1-p0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    # apply meanshift to get the new location
    # print(dst,old_roi,term_crit)
    ret, new_roi = cv2.meanShift(dst, old_roi, term_crit)
    # Draw it on image
    x,y,w,h = new_roi


    # # Select good points
    # good_new = p1[st==1]
    # good_old = p0[st==1]

    # # draw the tracks
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     #print i
    #     #print color[i]
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     cv2.circle(frame,(a, b),5,color[i].tolist(),-1)
    #     if i == 99:
    #         break


    # print("meas", meas, "cov",np.errorCovPre, "np.")
    # print(s,"state")    
    # print(meas,"meas")
    # print(center,"center",x_sp,y_sp,"speed")
    # print(good_new,good_old)
    # print (x, old_roi)
    if old_roi == new_roi:
        print("no update")
    else:
        ###### calculating speed and converting to tuples for showing on image
        # speed = good_new - good_old
        x_sp = x - old_roi[0]  #np.mean(speed[:,0])
        y_sp = y - old_roi[1] #np.mean(speed[:,1])
        # print(speed,x_sp,y_sp)
        x_new = x
        y_new = y
        x_old = old_roi[0]
        y_old = old_roi[1]

        m[0] = x
        m[1] = y
        m[2] = x_sp
        m[3] = y_sp
        s = kf.correct(m)
        print(s,'after update')

    # print(m,kf.transitionMatrix)
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
    img2 = cv2.rectangle(frame, (s[0],s[1]), (s[0]+r[2],s[1]+r[3]), 255,2)

    # err = kf.errorCovPost
    # radius = int(max(r[2],r[3]))
    # # print(radius)
    # mask_plot_ = cv2.circle(mask, center, radius, (255,0,0),1)
    # # mask_plot_ = cv2.rectangle(mask, (int(mask_plot[0]),int(mask_plot[1])),(int(mask_plot[0] + r[2]),int(mask_plot[1] + r[3])),(0,255,255),1)

    # img = cv2.add(frame,mask_plot_)
    # img = cv2.add(frame,mask)


    cv2.imshow('frame',img2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    # new_roi = old_roi.copy()
    # new_roi[0] = x
    # new_roi[1] = y
    old_roi = new_roi
    # p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()


