import numpy as np
import cv2
cap = cv2.VideoCapture(0)
# params for ShiTomasi corner detection

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
a = old_frame.shape



######### defining points
fact = 40
num_points = int(a[0]*a[1]/(fact**2))
points_to_track = np.zeros((num_points,1,2))
x = np.arange(a[0])
y = np.arange(a[1])
x = x[::fact]
y = y[::fact]

points_to_track[:,0,1] = np.repeat(x,int((a[1])/fact))
points_to_track[:,0,0] = np.tile(y,int((a[0])/fact))
p0 = points_to_track
p0 = points_to_track.astype(np.float32)


########## defining regions
height_ratio_y = 0.3
height_ratio_x = 0.3
center_ratio_y = 0.2
center_ratio_x = 0.3
avoid_ratio_x = 0.4
avoid_ratio_y = 0.2
region_height = [(int(height_ratio_x*a[1]),a[0] - int(height_ratio_y*a[0])),(a[1] - int(center_ratio_x*a[1]),a[0])]
region_center_1 = [(0,int(a[0]/2) - int(center_ratio_y*a[0]/2)),(int(center_ratio_x*a[1]),int(a[0]/2) + int(center_ratio_y*a[0]/2))]
region_center_2 = [(a[1] - int(center_ratio_x*a[1]),int(a[0]/2) - int(center_ratio_y*a[0]/2)),(a[1],int(a[0]/2) + int(center_ratio_y*a[0]/2))]
region_avoid = [(int(a[1]/2) - int(avoid_ratio_x*a[1]/2),int(a[0]/2) - int(avoid_ratio_y*a[0]/2)),(int(a[1]/2) + int(avoid_ratio_x*a[1]/2),int(a[0]/2) + int(avoid_ratio_y*a[0]/2))]




######## optical flow
while(1):
    ret,frame = cap.read()
    mask = np.zeros_like(old_frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    ###### calculating speed and converting to tuples for showing on image
    speed = p1-p0
    p1_ = p1[:,0,:]
    p1_ = tuple(map(tuple,p1_))
    p0_ = p0[:,0,:]
    p0_ = tuple(map(tuple,p0_))

    #### show regions and optical flow
    mask_height = cv2.rectangle(mask, region_height[0],region_height[1], (0,0,255), 1)
    mask_center_1 = cv2.rectangle(mask, region_center_1[0],region_center_1[1], (0,255,255), 1)
    mask_center_2 = cv2.rectangle(mask, region_center_2[0],region_center_2[1], (0,255,255), 1)	
    mask_avoid = cv2.rectangle(mask, region_avoid[0],region_avoid[1], (255,0,0), 1)	
    for i in range(len(p1)):
    	# print(p0_[i],p1_[i])
    	mask_of = cv2.arrowedLine(mask,p0_[i],p1_[i],(255,255,0),1)


    ############ optical flow in regions

    id_height = np.where(np.logical_and(np.logical_and(np.logical_and(p0[:,0,1]>=region_height[0][1] , p0[:,0,1]<=region_height[1][1]),p0[:,0,0]>=region_height[0][0]),p0[:,0,0]<=region_height[1][0]))
    id_height = id_height[0]
    of_height = speed[id_height,0,:]
    of_height_mean = [0.0,0.0]
    of_height_mean[0] = np.mean(of_height[0])
    of_height_mean[1] = np.mean(of_height[1])
    scale_mean_height_show = 20.0
    # print(of_height_mean)
    point_init = [int(a[1]/2),int((1-height_ratio_y/2)*a[0])]   
    mask_height_mean = cv2.arrowedLine(mask,(point_init[0],point_init[1]),(int(point_init[0]+scale_mean_height_show*of_height_mean[0]),int(point_init[1]+scale_mean_height_show*of_height_mean[1])),(255,255,0),3)
    



    id_center_1 = np.where(np.logical_and(np.logical_and(np.logical_and(p0[:,0,1]>=region_center_1[0][1] , p0[:,0,1]<=region_center_1[1][1]),p0[:,0,0]>=region_center_1[0][0]),p0[:,0,0]<=region_center_1[1][0]))
    id_center_1 = id_center_1[0]
    of_center_1 = speed[id_center_1,0,:]
    of_center_1_mean = [0.0,0.0]
    of_center_1_mean[0] = np.mean(of_center_1[0])
    of_center_1_mean[1] = np.mean(of_center_1[1])
    scale_mean_center_1_show = 20.0
    # print(of_height_mean)
    point_init = [int(center_ratio_x*a[1]/2),int(a[0]/2)]
    mask_center_1_mean = cv2.arrowedLine(mask,(point_init[0],point_init[1]),(int(point_init[0]+scale_mean_center_1_show*of_center_1_mean[0]),int(point_init[1]+scale_mean_center_1_show*of_center_1_mean[1])),(255,255,0),3)
    


    id_center_2 = np.where(np.logical_and(np.logical_and(np.logical_and(p0[:,0,1]>=region_center_2[0][1] , p0[:,0,1]<=region_center_2[1][1]),p0[:,0,0]>=region_center_2[0][0]),p0[:,0,0]<=region_center_2[1][0]))
    id_center_2 = id_center_2[0]
    of_center_2 = speed[id_center_2,0,:]
    of_center_2_mean = [0.0,0.0]
    of_center_2_mean[0] = np.mean(of_center_2[0])
    of_center_2_mean[1] = np.mean(of_center_2[1])
    scale_mean_center_2_show = 20.0
    # print(of_height_mean)
    point_init = [int(a[1]-center_ratio_x*a[1]/2),int(a[0]/2)]
    mask_center_2_mean = cv2.arrowedLine(mask,(point_init[0],point_init[1]),(int(point_init[0]+scale_mean_center_2_show*of_center_2_mean[0]),int(point_init[1]+scale_mean_center_2_show*of_center_2_mean[1])),(255,255,0),3)
    


    id_avoid = np.where(np.logical_and(np.logical_and(np.logical_and(p0[:,0,1]>=region_avoid[0][1] , p0[:,0,1]<=region_avoid[1][1]),p0[:,0,0]>=region_avoid[0][0]),p0[:,0,0]<=region_avoid[1][0]))
    id_avoid = id_avoid[0]
    of_avoid = speed[id_avoid,0,:]
    of_avoid_mean = [0.0,0.0]
    of_avoid_mean[0] = np.mean(of_avoid[0])
    of_avoid_mean[1] = np.mean(of_avoid[1])
    scale_mean_avoid_show = 20.0
    # print(of_height_mean)
    point_init = [int(a[1]/2),int(a[0]/2)]
    mask_avoid_mean = cv2.arrowedLine(mask,(point_init[0],point_init[1]),(int(point_init[0]+scale_mean_avoid_show*of_avoid_mean[0]),int(point_init[1]+scale_mean_avoid_show*of_avoid_mean[1])),(255,255,0),3)
    

    #### testing optical flow in  regions
    # id_height = id_avoid
    # for i in range(len(id_height)):
    #     p_temp = (int(p0_[id_height[i]][0]+20), int(p0_[id_height[i]][1]+10))
    #     mask_add = cv2.arrowedLine(mask,p0_[id_height[i]],p_temp,(0,0,255),2)


    #### adding all this on image for visualisation
    img = cv2.add(frame,mask_height_mean)
    img = cv2.add(frame,mask_center_1_mean)
    img = cv2.add(frame,mask_center_2_mean)
    img = cv2.add(frame,mask_avoid_mean)
    img = cv2.add(frame,mask_height)
    img = cv2.add(frame,mask_center_1)
    img = cv2.add(frame,mask_center_2)
    img = cv2.add(frame,mask_avoid)
    img = cv2.add(frame,mask_of)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
cv2.destroyAllWindows()
# cap.release()