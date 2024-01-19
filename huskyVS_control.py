# Make sure to have the add-on "ZMQ remote API" running in
# CoppeliaSim and have following scene loaded:
#
# scenes/messaging/synchronousImageTransmissionViaRemoteApi.ttt
#
# Do not launch simulation, but run this script
#
# All CoppeliaSim commands will run in blocking mode (block
# until a reply from CoppeliaSim is received). For a non-
# blocking example, see simpleTest-nonBlocking.py

import time

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import savetxt
from numpy.linalg import inv
#from matplotlib.animation import FuncAnimation

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print('Program started')



client = RemoteAPIClient('localhost',23004)
sim = client.getObject('sim')

visionSensorHandle = sim.getObject('/Vision_sensor')
fl_w = sim.getObject('/flw')
fr_w = sim.getObject('/frw')
rr_w = sim.getObject('/rrw')
rl_w = sim.getObject('/rlw')
IMU = sim.getObject('/Accelerometer_forceSensor')
#COM = sim.getObject('/Husky/ReferenceFrame')
COM = sim.getObject('/Husky/Accelerometer/Accelerometer_mass')
Husky_ref = sim.getObject('/Husky')
#InertialFrame = sim.getObject('/InertialFrame')

#Gyro = sim.getObject('/GyroSensor_reference')
#gyroCommunicationTube=sim.tubeOpen(0,'gyroData'..sim.getNameSuffix(nil),1)


x_acc = []
y_acc = []
z_acc = []
x_ang = []
y_ang = []
z_ang = []
x_pos = []
y_pos = []
lin_vel = []
ang_vel = []
lin_vel2 = []
lin_vel3 = []
ang_vel2 = []
sim_time = []
cmd_wheel_l = []
cmd_wheel_r = []
wheel_l = []
wheel_r = []
rlz_wheel_l = []
rlz_wheel_r = []
counter = []


# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Run a simulation in stepping mode:
client.setStepping(True)
sim.startSimulation()

'''
#sim.setFloatSignal("friction_coeff",3)
data = np.genfromtxt('/home/asalvi/code_workspace/Husky_CS_SB3/control/data_random.csv', delimiter=',')
#print(data)
print(np.shape(data))

#import csv

#with open('/home/asalvi/code_workspace/Husky_CS_SB3/control/data_random.csv', newline='') as f:
#    reader = csv.reader(f)
#    data = list(reader)


#print(data)

'''

while (t:= sim.getSimulationTime()) < 600:
    #print(t)
    
    # IMAGE PROCESSING CODE ################
    '''
    img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)

    # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
    # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
    # and color format is RGB triplets, whereas OpenCV uses BGR:
    
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0)
    cv2.imshow('', img)
    # Cropping an image
    cropped_image = img[175:225, 0:256]
    # Display cropped image
    cv2.imshow("cropped", cropped_image)

    '''

    img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
            # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
            # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
            # and color format is RGB triplets, whereas OpenCV uses BGR:
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale

            # Current image
    cropped_image = img[288:480, 192:448] # Crop image to only to relevant path data (Done heuristically)
    im_bw = cv2.threshold(cropped_image, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    #print(im_bw.shape)
 
    # calculate moments of binary image
    M = cv2.moments(im_bw)
 
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    resize_img = cv2.resize(cropped_image  , (10 , 5))
    cv2.imshow("Low Res", resize_img)
    print(resize_img.shape)

    print(cX)
 
    # put text and highlight the center
    cv2.circle(cropped_image, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(cropped_image, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

 
    # display the image
    cv2.imshow("Image", cropped_image)

    cv2.waitKey(1)

    ###################### Image processing Ends here

    # Centering Error :
    error = 128 - cX

    ###### Generate control commands

    p_gain = -0.05

    V = 0.3
    omega = p_gain*error 

    ## Control map:
    # x_dot = [0.081675 0.081675; -0.1081 -0.1081] phi_dot

    A = np.array([[0.081675,0.081675],[-0.1081,0.1081]]) 
    velocity = np.array([[V],[omega]])
    phi_dots = np.matmul(inv(A),velocity)

    # INPUT
    # WHEEL CONTROL CODE 3.391 6.056
    #Left_ = 3
    #Right_ = 3 + 0.5*np.sin(0.3*t + 0)
    #cmd_wheel_l.append(Left_)
    #cmd_wheel_r.append(Right_)



    
    # Convert to V and Omega
    #A_mat = np.array([[0.0825,0.0825],[-0.2775,0.27755]])
    #wheel_vels = np.array([[Left_],[Right_]])
    #print(wheel_vels)
    #velocity = np.matmul(A_mat,wheel_vels)
    #print(velocity)
    #A_inv = np.array([[6.1358,-2.9177],[6.1358,2.9177]])
    #phi_dots = np.matmul(A_inv,velocity)
    #print(phi_dots)
    phi_dots = phi_dots.astype(float)
    Left = phi_dots[0].item()
    Right = phi_dots[1].item()
    


    sim.setJointTargetVelocity(fl_w, Left)
    sim.setJointTargetVelocity(fr_w, Right)
    sim.setJointTargetVelocity(rl_w, Left)
    sim.setJointTargetVelocity(rr_w, Right)
    wheel_l.append(Left)
    wheel_r.append(Right)
    

    # IMU and Gyro readings
    IMU_X = sim.getFloatSignal("myIMUData_X")
    if IMU_X:
        x_acc.append(IMU_X)
        #print(IMU_X)
    IMU_Y = sim.getFloatSignal("myIMUData_Y")
    if IMU_Y:
        y_acc.append(IMU_Y)
        #print(IMU_Y)
    IMU_Z = sim.getFloatSignal("myIMUData_Z")
    if IMU_Z:
        z_acc.append(IMU_Z)
        #print(IMU_Z)

    Gyro_X = sim.getFloatSignal("myGyroData_angX")
    if IMU_X:
        x_ang.append(Gyro_X)
        #print(Gyro_X)
    Gyro_Y = sim.getFloatSignal("myGyroData_angY")
    if Gyro_Y:
        y_ang.append(Gyro_Y)
        #print(Gyro_Y)
    Gyro_Z = sim.getFloatSignal("myGyroData_angZ")
    if Gyro_Z:
        z_ang.append(Gyro_Z)
        #print(Gyro_Z)


    # Realized joint velocity
    rlz_l = sim.getJointVelocity(fl_w)
    rlz_r = sim.getJointVelocity(fr_w)
    rlz_wheel_l.append(rlz_l)
    rlz_wheel_r.append(rlz_r)


    sim_time.append(t)
    print(t)

    client.step()  # triggers next simulation step



sim.stopSimulation()

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

#cv2.destroyAllWindows()

print('Program ended')

#savetxt('x_acc_cc.csv', x_acc, delimiter=',')
#savetxt('y_acc_cc.csv', y_acc, delimiter=',')
#savetxt('z_acc_cc.csv', z_acc, delimiter=',')
#savetxt('x_ang_cc.csv', x_ang, delimiter=',')
#savetxt('y_ang_cc.csv', y_ang, delimiter=',')
#savetxt('z_ang_06_05.csv', z_ang, delimiter=',')

#def every_tenth(data,no):
    #data2 = data[no - 1::no]
    #return [data[0],data2]

#print(len(lin_vel))
#lin_vel = every_tenth(lin_vel,10)
#print(len(lin_vel))



savetxt('x_pos_.csv', x_pos, delimiter=',')
savetxt('y_pos_.csv', y_pos, delimiter=',')
savetxt('x_vel.csv', lin_vel, delimiter=',')
savetxt('x_vel2.csv', lin_vel2, delimiter=',')
#savetxt('x_vel3.csv', lin_vel3, delimiter=',')
savetxt('z_ang.csv', z_ang, delimiter=',')
savetxt('z_ang2.csv', ang_vel2, delimiter=',') # Angular from conversion (Inertial to body)
savetxt('wheel_l.csv', wheel_l, delimiter=',')
savetxt('wheel_r.csv', wheel_r, delimiter=',')
savetxt('rlz_wheel_l.csv', rlz_wheel_l, delimiter=',')
savetxt('rlz_wheel_r.csv', rlz_wheel_r, delimiter=',')
savetxt('cmd_wheel_l.csv', cmd_wheel_l, delimiter=',')
savetxt('cmd_wheel_r.csv', cmd_wheel_r, delimiter=',')


print('Done saving files')