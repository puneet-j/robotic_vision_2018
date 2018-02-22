# This is for problems with clashing opencv versions from ROS installations
import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

import cv2
import numpy as np
import math
import time
# import pygame
from uav_sim import UAVSim
# from multi_image import MultiImage

urban_world = 'UrbanCity'
forest_world = 'EuropeanForest'
redwood_world = 'RedwoodForest'


def holodeck_sim():
    global of_height_mean
    uav_sim = UAVSim(urban_world)
    print("ive reached here")
    uav_sim.init_teleop()
    # uav_sim.init_plots(plotting_freq=5) # Commenting this line would disable plotting
    uav_sim.command_velocity = True # This tells the teleop to command velocities rather than angles

    uav_sim.step_sim()
    cap = uav_sim.get_camera()
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (25,25),
                      maxLevel = 10,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 0.06))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # ret, old_frame = cap.read()
    old_frame = cap
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    a = old_frame.shape



    ######### defining points
    fact = 20
    x = np.arange(a[0])
    y = np.arange(a[1])
    x = x[::fact]
    y = y[::fact]
    num_points = len(x)*len(y)
    points_to_track = np.zeros((num_points,1,2))

    print(num_points,"num points")
    print(len(x),len(y),"lengths")
    print(int(a[1]/fact),int(a[0]/fact),"length of points to track")
    points_to_track[:,0,1] = np.repeat(x,len(x))
    points_to_track[:,0,0] = np.tile(y,len(y))
    p0 = points_to_track
    p0 = points_to_track.astype(np.float32)


    ########## defining regions
    height_ratio_y = 0.3
    height_ratio_x = 0.3
    center_ratio_y = 0.2
    center_ratio_x = 0.3
    avoid_ratio_x = 0.6
    avoid_ratio_y = 0.2
    region_height = [(int(height_ratio_x*a[1]),a[0] - int(height_ratio_y*a[0])),(a[1] - int(center_ratio_x*a[1]),a[0])]
    region_center_1 = [(0,int(a[0]/2) - int(center_ratio_y*a[0]/2)),(int(center_ratio_x*a[1]),int(a[0]/2) + int(center_ratio_y*a[0]/2))]
    region_center_2 = [(a[1] - int(center_ratio_x*a[1]),int(a[0]/2) - int(center_ratio_y*a[0]/2)),(a[1],int(a[0]/2) + int(center_ratio_y*a[0]/2))]
    region_avoid = [(int(a[1]/2) - int(avoid_ratio_x*a[1]/2),int(a[0]/2) - int(avoid_ratio_y*a[0]/2)),(int(a[1]/2) + int(avoid_ratio_x*a[1]/2),int(a[0]/2) + int(avoid_ratio_y*a[0]/2))]



    edge_min = 150
    edge_max = 200

    # multi_img = MultiImage(2,2)

    while True:
        # This is the main loop where the simulation is updated
        uav_sim.step_sim()
        cam = uav_sim.get_camera()

        # I run my opencv stuff here
        gray = cv2.cvtColor(cam, cv2.COLOR_RGBA2GRAY)
        edge = cv2.Canny(cam, edge_min, edge_max)
        bgr = cv2.cvtColor(cam, cv2.COLOR_RGBA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # You can also give an external command and switch between automatic
        # commands given here and manual commands from the keyboard using the
        # key mapped to MANUAL_TOGGLE in uav_sim.py

        # In automatic mode, fly forward at 3m altitude at current heading
        # print(uav_sim)
        # yaw_c = uav_sim.yaw_c
        # uav_sim.command_velocity(vx=2.0, vy=0.0, yaw=yaw_c, 3.0)

        # This is just a useful class for viewing multiple filters in one image
        # multi_img.add_image(cam, 0,0)
        # multi_img.add_image(gray, 0,1)
        # multi_img.add_image(edge, 1,0)
        # multi_img.add_image(hsv, 1,1)
        # display = multi_img.get_display()
        # display = gray.get_display()
        frame = cam
        # print(frame.shape)

        mask = np.zeros_like(old_frame)
        frame_gray = gray
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
        of_height_mean[0] = np.mean(of_height[:,0])
        of_height_mean[1] = np.mean(of_height[:,1])
        scale_mean_height_show = 5.0
        # print(of_height_mean)
        point_init = [int(a[1]/2),int((1-height_ratio_y/2)*a[0])]
        mask_height_mean = cv2.arrowedLine(mask,(point_init[0],point_init[1]),(int(point_init[0]+scale_mean_height_show*of_height_mean[0]),int(point_init[1]+scale_mean_height_show*of_height_mean[1])),(255,255,0),2)
        # print(of_height_mean)#of_height_mean[1]==3.5


        id_center_1 = np.where(np.logical_and(np.logical_and(np.logical_and(p0[:,0,1]>=region_center_1[0][1] , p0[:,0,1]<=region_center_1[1][1]),p0[:,0,0]>=region_center_1[0][0]),p0[:,0,0]<=region_center_1[1][0]))
        id_center_1 = id_center_1[0]
        of_center_1 = speed[id_center_1,0,:]
        of_center_1_mean = [0.0,0.0]
        of_center_1_mean[0] = np.mean(of_center_1[:,0])
        of_center_1_mean[1] = np.mean(of_center_1[:,1])
        scale_mean_center_1_show = 5.0
        # print(of_height_mean)
        point_init = [int(center_ratio_x*a[1]/2),int(a[0]/2)]
        mask_center_1_mean = cv2.arrowedLine(mask,(point_init[0],point_init[1]),(int(point_init[0]+scale_mean_center_1_show*of_center_1_mean[0]),int(point_init[1]+scale_mean_center_1_show*of_center_1_mean[1])),(255,255,0),2)


        id_center_2 = np.where(np.logical_and(np.logical_and(np.logical_and(p0[:,0,1]>=region_center_2[0][1] , p0[:,0,1]<=region_center_2[1][1]),p0[:,0,0]>=region_center_2[0][0]),p0[:,0,0]<=region_center_2[1][0]))
        id_center_2 = id_center_2[0]
        of_center_2 = speed[id_center_2,0,:]
        of_center_2_mean = [0.0,0.0]
        of_center_2_mean[0] = np.mean(of_center_2[:,0])
        of_center_2_mean[1] = np.mean(of_center_2[:,1])
        scale_mean_center_2_show = 5.0
        # print(of_height_mean)
        point_init = [int(a[1]-center_ratio_x*a[1]/2),int(a[0]/2)]
        mask_center_2_mean = cv2.arrowedLine(mask,(point_init[0],point_init[1]),(int(point_init[0]+scale_mean_center_2_show*of_center_2_mean[0]),int(point_init[1]+scale_mean_center_2_show*of_center_2_mean[1])),(255,255,0),2)


    # region_height = [(int(height_ratio_x*a[1]),a[0] - int(height_ratio_y*a[0])),(a[1] - int(center_ratio_x*a[1]),a[0])]
    # region_center_1 = [(0,int(a[0]/2) - int(center_ratio_y*a[0]/2)),(int(center_ratio_x*a[1]),int(a[0]/2) + int(center_ratio_y*a[0]/2))]
    # region_center_2 = [(a[1] - int(center_ratio_x*a[1]),int(a[0]/2) - int(center_ratio_y*a[0]/2)),(a[1],int(a[0]/2) + int(center_ratio_y*a[0]/2))]
    # region_avoid = [(int(a[1]/2) - int(avoid_ratio_x*a[1]/2),int(a[0]/2) - int(avoid_ratio_y*a[0]/2)),(int(a[1]/2) + int(avoid_ratio_x*a[1]/2),int(a[0]/2) + int(avoid_ratio_y*a[0]/2))]





        id_avoid_1 = np.where(np.logical_and(np.logical_and(np.logical_and(p0[:,0,1]>=int(a[0]/2) , p0[:,0,1]<=region_avoid[1][1]),p0[:,0,0]>=region_avoid[0][0]),p0[:,0,0]<=region_avoid[1][0]))
        id_avoid_1 = id_avoid_1[0]
        of_avoid_1 = speed[id_avoid_1,0,:]
        of_avoid_1_mean = [0.0,0.0]
        of_avoid_1_mean[0] = np.mean(of_avoid_1[:,0])
        of_avoid_1_mean[1] = np.mean(of_avoid_1[:,1])

        id_avoid = np.where(np.logical_and(np.logical_and(np.logical_and(p0[:,0,1]>=region_avoid[0][1] , p0[:,0,1]<=int(a[0]/2)),p0[:,0,0]>=region_avoid[0][0]),p0[:,0,0]<=region_avoid[1][0]))
        id_avoid = id_avoid[0]
        of_avoid = speed[id_avoid,0,:]
        of_avoid_mean = [0.0,0.0]
        of_avoid_mean[0] = np.mean(of_avoid[:,0])
        of_avoid_mean[1] = np.mean(of_avoid[:,1])

        scale_mean_avoid_show = 5.0
        # print(of_height_mean)
        point_init = [int(a[1]/2),int(a[0]/2)]
        mask_avoid_mean = cv2.arrowedLine(mask,(point_init[0],point_init[1]),(int(point_init[0]+scale_mean_avoid_show*of_avoid_mean[0]),int(point_init[1]+scale_mean_avoid_show*of_avoid_mean[1])),(255,255,0),2)


        mode = uav_sim.get_follow_mode()
        if uav_sim.get_mode()==1:
            if mode%3 == 0:
                uav_sim.automatic_control(of_height_mean,1)
                print("Height mode")
            elif mode%3 == 1:
                uav_sim.automatic_control([of_center_1_mean, of_center_2_mean],2)
                print("center mode")
            elif mode%3 == 2:
                uav_sim.automatic_control([of_avoid_mean, of_avoid_1_mean],3)
                print("avoid mode")

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

        cv2.imshow('Holodeck', img)
        cv2.waitKey(1)
        old_gray = frame_gray.copy()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    holodeck_sim()
    print("Finished")
