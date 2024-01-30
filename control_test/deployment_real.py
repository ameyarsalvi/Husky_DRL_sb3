

import time

import math
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
from numpy import savetxt
from numpy.linalg import inv
import stable_baselines3
from stable_baselines3 import PPO
import torch
# For inference

#from torch import Model # Made up package
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PPO.load("/home/asalvi/Downloads/logs_br/log_single/best_model_parallel_VS.zip", device='cuda')

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print('Program started')



client = RemoteAPIClient('localhost',23002)
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


while (t:= sim.getSimulationTime()) < 600:
    #print(t)
    
    # IMAGE PROCESSING CODE ################


    img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale
    cropped_image = img[288:480, 192:448] # Crop image to only to relevant path data (Done heuristically)
    crop_error = img[400:480, 192:448] # Crop image to only to relevant path data (Done heuristically)
    im_bw = cv2.threshold(cropped_image, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
    noisy_image = cv2.add(im_bw, noise)
    im_bw = np.frombuffer(noisy_image, dtype=np.uint8).reshape(192, 256, 1) # Reshape to required observation size
    im_bw_obs = ~im_bw

    #Calcuation of centroid for lane centering
    crop_error = ~crop_error
    M = cv2.moments(crop_error) # calculate moments of binary image
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    #cv2.imshow("Image", crop_error)
    #cv2.waitKey(1)
        

    # Centering Error :
    error = 128 - cX #Lane centering error that can be used in reward function

    

    #Neural network for inference ############### << Here

    noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
    noisy_image = cv2.add(im_bw, noise)
    im_bw = np.frombuffer(noisy_image, dtype=np.uint8).reshape(192, 256, 1)
    im_bw =~im_bw
    inputs = np.array(im_bw,dtype = np.uint8)

    observation_size = model.observation_space.shape


    # Run forward pass

    pred = model.policy.predict(inputs)
    a= pred[0]
    V = 0.25*a[0].item() + 0.75 # >> Constrain to [0.6 0.7] >> Complete space 0 -> 1  
    omega = 0.6*a[1].item()#Omega range : [-0.5 0.5]

    ##################### <<< To Here

    
    ###### Generate control commands
    '''
    p_gain = -0.01

    V = 0.5
    omega = p_gain*error 
    print(error)
    #print()
    print(omega)
    #print(omega.type)
    '''
    
    ## Control map:
    # x_dot = [0.081675 0.081675; -0.1081 -0.1081] phi_dot
    t_a = 0.7510
    t_b = 1.5818

        
    A = np.array([[t_a*0.0825,t_a*0.0825],[-0.1486/t_b,0.1486/t_b]])

    #A = np.array([[0.081675,0.081675],[-0.1081,0.1081]]) 
    velocity = np.array([[V],[omega]])
    phi_dots = np.matmul(inv(A),velocity)

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
    #print(t)

    client.step()  # triggers next simulation step



sim.stopSimulation()

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

#cv2.destroyAllWindows()

print('Program ended')

#savetxt('x_acc_cc.csv', x_acc, delimiter=',')

print('Done saving files')