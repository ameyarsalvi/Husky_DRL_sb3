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

#model = PPO.load("/home/asalvi/Downloads/logs_br/log_ten/best_model_parallel_VS.zip", device='cuda')
#print(model.policy)

#model = PPO()
#model.load_state_dict(torch.load("home/asalvi/code_workspace/Husky_CS_SB3/train/test/policy.pth"))

#model = model.to(device) # Set model to gpu
#model.eval()

#from matplotlib.animation import FuncAnimation

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print('Program started')



client = RemoteAPIClient('localhost',23006)
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
CameraJoint = sim.getObject('/FORBody/Husky/Camera_joint')
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
cam_err_old = 0


### camera controller ##



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
img_no = 0

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
    #cv2.imwrite('/home/asalvi/code_workspace/tmp/image_data/raw_img.png', img) 
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale
    #cv2.imwrite('/home/asalvi/code_workspace/tmp/image_data/bw_img.png', img) 

    # Current image
    #cropped_image = img[270:480, 0:640] # Crop image to only to relevant path data (Done heuristically)
    cropped_image = img[270:300, 0:640] # Crop image to only to relevant path data (Done heuristically)
    #cropped_image = img[350:400, 0:640] # Crop image to only to relevant path data (Done heuristically)
    #cropped_image = img[288:480, 0:640] # Crop image to only to relevant path data (Done heuristically)
    cropped_image = cv2.resize(cropped_image, (0,0), fx=0.5, fy=0.5)
    im_bw = cv2.threshold(cropped_image, 125, 200, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    im_bw = cv2.threshold(im_bw, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
    noisy_image = cv2.add(im_bw, noise)
    k = ~noisy_image
    #im_bw = np.frombuffer(noisy_image, dtype=np.uint8).reshape(96, 320, 1) # Reshape to required observation size
    #obs = ~im_bw


    # calculate moments of binary image
    M = cv2.moments(k)
 
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    #resize_img = cv2.resize(cropped_image  , (10 , 5))
    #cv2.imshow("Low Res", resize_img)
    #print(resize_img.shape)

    print(cX)
    print(cY)
 
    # put text and highlight the center
    cv2.circle(k, (cX, cY), 5, (255, 255, 255), 1)
    cv2.putText(k, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cropped_image2 = img[270:480, 0:640] # Crop image to only to relevant path data (Done heuristically)
    cropped_image2 = cv2.resize(cropped_image2, (0,0), fx=0.5, fy=0.5)
    im_bw2 = cv2.threshold(cropped_image2, 125, 200, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    im_bw2 = cv2.threshold(im_bw2, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    noise2 = np.random.normal(0, 25, im_bw2.shape).astype(np.uint8)
    noisy_image2 = cv2.add(im_bw2, noise2)
    im_bw2 = np.frombuffer(noisy_image2, dtype=np.uint8).reshape(105, 320,1) # Reshape to required observation size
    obs2 = ~im_bw2
    
 
    

    cv2.imshow("Image", cropped_image)
    cv2.imshow("NN Input", obs2)
    cv2.imshow("ImageProc", k)
    #cv2.imshow("ImageProcCent", noisy_image)
    #cv2.imshow("Image", im_bw)

    cv2.waitKey(1)

    #im_bw = cv2.threshold(cropped_image, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    #cv2.imwrite('/home/asalvi/code_workspace/tmp/image_data/AEImg/'+ str(img_no) + '.png', obs2) 
    '''

    noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
    noisy_image = cv2.add(im_bw, noise)
    cv2.imwrite('/home/asalvi/code_workspace/tmp/image_data/noise.png', noisy_image) 
    #print(im_bw.shape)
 
    # calculate moments of binary image
    M = cv2.moments(im_bw)
 
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    #resize_img = cv2.resize(cropped_image  , (10 , 5))
    #cv2.imshow("Low Res", resize_img)
    #print(resize_img.shape)

    #print(cX)
 
    # put text and highlight the center
    cv2.circle(im_bw, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(im_bw, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

 
    # display the image
    cv2.imshow("Image", im_bw)

    cv2.waitKey(1)
    '''

    ###################### Image processing Ends here

    ####### Camera Position Control

    CamPosition = sim.getJointPosition(CameraJoint)
    print(CamPosition)

    cam_error = (160-cX)
    CamFeed = 0.01*cam_error - 0.005*(np.abs(cam_error-cam_err_old))
    camSet = np.clip(CamFeed*(3.14/180),-3.14/4,3.14/4)
    cam_err_old = cam_error

    #sim.setJointPosition(CameraJoint, CamPosition +  camSet.item())

    #CamPosition = sim.getJointPosition(CameraJoint)
    #print(CamPosition)

    ########## Motion control part


    V_lin = 1.0
    Omg_ang = 0.01*cam_error
        #camActSet = self.camAngle

        # No condition
    t_a = 0.7510
    t_b = 1.5818
        
        
        
    A = np.array([[t_a*0.0825,t_a*0.0825],[-0.1486/t_b,0.1486/t_b]])
    velocity = np.array([V_lin,Omg_ang])
    phi_dots = np.matmul(inv(A),velocity) #Inverse Kinematics
    phi_dots = phi_dots.astype(float)
    Left = phi_dots[0].item()
    Right = phi_dots[1].item()

    sim.setJointTargetVelocity(fl_w, Left)
    sim.setJointTargetVelocity(fr_w, Right)
    sim.setJointTargetVelocity(rl_w, Left)
    sim.setJointTargetVelocity(rr_w, Right)
    wheel_l.append(0)
    wheel_r.append(0)
    

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
    img_no += 1



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


'''
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
'''

print('Done saving files')


# Centering Error :
    #error = float(128 - cX)

    

    #Neural network for inference ############### << Here
'''
    noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
    noisy_image = cv2.add(im_bw, noise)
    im_bw = np.frombuffer(noisy_image, dtype=np.uint8).reshape(192, 256, 1)
    im_bw =~im_bw
    inputs = np.array(im_bw,dtype = np.uint8)
    #inputs = torch.from_numpy(inputs)
    #torch.reshape(inputs, (1, 192, 256, 1))
    #inputs = inputs.view(1, -1)
    #inputs = inputs.float64()
    #print(inputs.shape)
    observation_size = model.observation_space.shape
    #print(observation_size)
    #inputs = inputs.to(device) # You can move your input to gpu, torch defaults to cpu
    
    #inputs = stable_baselines3.common.utils.obs_as_tensor(inputs, device)
    #inputs = inputs.unsqueeze(0)
'''

'''

    # Run forward pass
    #with torch.no_grad():
    pred = model.policy.predict(inputs)
    #print(pred)
    #print(pred[0])
    a= pred[0]
    #print(a.type)
    #print(a[0])
    #V = pred[0]
    V = 0.25*a[0].item() + 0.75 # >> Constrain to [0.6 0.7] >> Complete space 0 -> 1  
    #print(V)
    #omega = pred[1]
    omega = 0.6*a[1].item()#Omega range : [-0.5 0.5]
    #print(V)
    #print(omega)
    
    ##################### <<< To Here

    
    ###### Generate control commands
    '''
    #p_gain = -0.01

    #V = 0.5
    #omega = p_gain*error 
    #print(error)
    #print()
    #print(omega)
    #print(omega.type)
'''
    
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
    
'''