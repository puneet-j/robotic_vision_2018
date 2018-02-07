#!/usr/bin/env python

from __future__ import absolute_import, division, unicode_literals, print_function
import tty, termios
import sys
import _thread
import time
from Holodeck import Holodeck
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

env = Holodeck.make("EuropeanForest")

def getch():   # define non-Windows version
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def keypress():
    global char
    char = getch()

def main():
    global char,up,roll,pitch,yaw,setting,location,vel,imu,orient
    location = [[0.,0.,0.]]
    vel = [[0.,0.,0.]]
    imu = [[0.,0.,0.,0.,0.,0.]]
    orient = [[0.,0.,0.]]
    # env.reset()
    cv2.namedWindow("Holodeck",cv2.WINDOW_OPENGL)
    up = 0.0
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    setting = 0
    keep_running()
    char = None
    _thread.start_new_thread(keypress, ())

    while True:
        if char is not None:
            try:
                print("Key pressed is " + char)
                do_something()
            except UnicodeDecodeError:
                print("character can not be decoded, sorry!")
                char = None
            _thread.start_new_thread(keypress, ())
            if char == 'q' or char == '\x1b':  # x1b is ESC
                cv2.destroyAllWindows()
                plot_everything()
                time.sleep(2)
                plt.close()
                exit()
            char = None
        #print("Program is running")
        keep_running()
        #time.sleep(0.01)
def plot_everything():
    global location,vel,imu,orient
    # print(location)
    # print(location.shape)
    plt.figure(1)
    plt.subplot(311)
    plt.plot(location[2:,0])
    plt.title('location')
    plt.ylabel('north')
    plt.subplot(312)
    plt.plot(location[2:,1])
    plt.ylabel('west')
    plt.subplot(313)
    plt.plot(location[2:,2])
    plt.ylabel('up')
    plt.xlabel('time')
    plt.show()

    plt.figure(2)
    plt.subplot(311)
    plt.plot(orient[2:,0])
    plt.title('orient')
    plt.ylabel('phi')
    plt.subplot(312)
    plt.plot(orient[2:,1])
    plt.ylabel('theta')
    plt.subplot(313)
    plt.plot(-orient[2:,2])
    plt.ylabel('psi')
    plt.xlabel('time')
    plt.show()

    plt.figure(3)
    plt.subplot(311)
    plt.plot(vel[2:,0])
    plt.title('vel')
    plt.ylabel('north')
    plt.subplot(312)
    plt.plot(vel[2:,1])
    plt.ylabel('west')
    plt.subplot(313)
    plt.plot(vel[2:,2])
    plt.ylabel('up')
    plt.xlabel('time')
    plt.show()

    plt.figure(4)
    plt.subplot(321)
    plt.plot(imu[2:,0])
    plt.title('imu')
    plt.ylabel('a_x')
    plt.subplot(322)
    plt.plot(imu[2:,1])
    plt.ylabel('a_y')
    plt.subplot(323)
    plt.plot(imu[2:,2])
    plt.ylabel('a_z')
    plt.subplot(324)
    plt.plot(imu[2:,3])
    plt.ylabel('\omega_x')
    plt.subplot(325)
    plt.plot(imu[2:,4])
    plt.ylabel('\omega_y')
    plt.subplot(326)
    plt.plot(imu[2:,5])
    plt.ylabel('\omega_z')
    plt.xlabel('time')
    plt.show()

def keep_running():
    global roll,pitch,yaw,up,location,vel,imu,orient,location,vel,imu,orient
    state, reward, terminal, _  =  env.step(np.copy([roll,pitch,yaw,up]))
    frame = np.copy(state[Sensors.PRIMARY_PLAYER_CAMERA])
    imu = np.append(imu,np.transpose(np.copy(state[Sensors.IMU_SENSOR])),axis=0)
    # print(orient)
    # print(np.transpose(np.copy(mat2euler(state[Sensors.ORIENTATION_SENSOR],'sxyz'))))
    orient = np.append(orient,[np.transpose(np.copy(mat2euler(state[Sensors.ORIENTATION_SENSOR],'sxyz')))],axis=0)
    # orient = np.append(imu,np.copy(state[Sensors.IMU_SENSOR]),axis=0)
    # print(location)
    # print(np.transpose(np.copy(state[Sensors.LOCATION_SENSOR])))
    location = np.append(location,np.transpose(np.copy(state[Sensors.LOCATION_SENSOR])),axis=0)
    vel = np.append(vel,np.transpose(np.copy(state[Sensors.VELOCITY_SENSOR])),axis=0)
    filtered(frame)
    #print("Im running, state",state[Sensors.IMU_SENSOR],"\n")
    #print("Command \n",[roll,pitch,yaw,up])
    #print("\n")
    # time.sleep(0.03)

def filtered(frame):
    global setting
    if setting == '1':
        #print("sobel now")
        frame = cv2.Sobel(frame,-1,0,1,100)
    elif setting == '2':
        # print("canny now")
        frame = cv2.Sobel(frame,-1,1,0,100)
    elif setting == '3':
        # print("canny now")
        frame = cv2.Sobel(frame,-1,1,1,100)
    elif setting == '4':
        # print("canny now")
        frame = cv2.filter2D(frame,cv2.CV_32F,np.matrix([[0,-1,0],[-1,5,-1],[0,-1,0]]))
    elif setting == '5':
        # print("canny now")
        frame = cv2.blur(frame,(5,5))
    else:
        frame = frame
    frame = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
    # print(frame.shape)
    cv2.imshow("Holodeck",frame)
    cv2.waitKey(10)
    # time.sleep(0.03)
    # return frame

def do_something():
    global roll,pitch,yaw,up,char,setting
    print (char,up,yaw,pitch,roll)
    if char == 'i':
        up += 0.4
    elif char == 'k':
        up -= 0.4
    elif char == 'l':
        yaw -= 0.1
    elif char == 'j':
        yaw += 0.1
    elif char == 'w':
        pitch += 0.1
    elif char == 's':
        pitch -= 0.1
    elif char == 'd':
        roll -= 0.1
    elif char == 'a':
        roll += 0.1
    else:
        setting = char

if __name__ == "__main__":
    main()
