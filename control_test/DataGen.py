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
#import cv2
import matplotlib.pyplot as plt
from numpy import savetxt
#from matplotlib.animation import FuncAnimation

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print('Program started')



client = RemoteAPIClient('localhost',23002)
sim = client.getObject('sim')

#ctrlPts = [x y z qx qy qz qw]

#pathHandle = sim.createPath(ctrlPts, options = 0, subdiv = 100, smoothness = 1.0, orientationMode = 0, upVector = [0, 0, 1])



visionSensorHandle = sim.getObject('/Vision_sensor')
fl_w = sim.getObject('/flw')
fr_w = sim.getObject('/frw')
rr_w = sim.getObject('/rrw')
rl_w = sim.getObject('/rlw')
IMU = sim.getObject('/Accelerometer_forceSensor')
#COM = sim.getObject('/Husky/ReferenceFrame')
COM = sim.getObject('/Husky/Accelerometer/Accelerometer_mass')
Husky_ref = sim.getObject('/Husky')
flw_fric = sim.getObject('/Husky/front_left_wheel_link_respondable')
frw_fric = sim.getObject('/Husky/front_right_wheel_link_respondable')
rlw_fric = sim.getObject('/Husky/rear_left_wheel_link_respondable')
rrw_fric = sim.getObject('/Husky/rear_right_wheel_link_respondable')
floor_fric = sim.getObject('/Floor')

#InertialFrame = sim.getObject('/InertialFrame')

#Gyro = sim.getObject('/GyroSensor_reference')
#gyroCommunicationTube=sim.tubeOpen(0,'gyroData'..sim.getNameSuffix(nil),1)

mass = 100
friction = 1
i_zz = 0

sim.setShapeMass(Husky_ref, mass)
sim.setEngineFloatParam(sim.mujoco_body_friction1, flw_fric, friction)
sim.setEngineFloatParam(sim.mujoco_body_friction1, frw_fric, friction)
sim.setEngineFloatParam(sim.mujoco_body_friction1, rlw_fric, friction)
sim.setEngineFloatParam(sim.mujoco_body_friction1, rrw_fric, friction)
sim.setEngineFloatParam(sim.mujoco_body_friction1, floor_fric, friction)

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


#sim.setFloatSignal("friction_coeff",3)
#data = np.genfromtxt('/home/asalvi/code_workspace/Husky_CS_SB3/control/data_random.csv', delimiter=',')



while (t:= sim.getSimulationTime()) < 60:
    
    # INPUT
    Left_ = 3
    Right_ = 3 + 0.5*np.sin(0.3*t + 0)
    cmd_wheel_l.append(Left_)
    cmd_wheel_r.append(Right_)

    sim.setJointTargetVelocity(fl_w, Left_)
    sim.setJointTargetVelocity(fr_w, Right_)
    sim.setJointTargetVelocity(rl_w, Left_)
    sim.setJointTargetVelocity(rr_w, Right_)


    #OUTPUT

    # Position for validation
    position = sim.getObjectPosition(Husky_ref,sim.handle_world)
    if position[0] == None:
        pass
    else:
        x_pos.append(position[0])
        y_pos.append(position[1])
        #print(position[0],position[1])

    # Linear Velocity for validation
    linear_vel, angular_vel = sim.getVelocity(Husky_ref)
    sRb = sim.getObjectMatrix(COM,sim.handle_world)
    Rot = np.array([[sRb[0],sRb[1],sRb[2]],[sRb[4],sRb[5],sRb[6]],[sRb[8],sRb[9],sRb[10]]])
    vel_body = np.matmul(np.transpose(Rot),np.array([[linear_vel[0]],[linear_vel[1]],[linear_vel[2]]]))
    realized_vel = np.abs(-1*vel_body[2].item())

    lin_vel.append(realized_vel)


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


    sim_time.append(t)
    #print("Time")
    print(t)

    client.step()  # triggers next simulation step



sim.stopSimulation()

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

#args = (lin_vel, z_ang, x_pos, y_pos)

data = np.vstack((lin_vel, z_ang,x_pos, y_pos))
np.savetxt("data.csv", data, delimiter=",")


print('Program ended')

#savetxt('/home/asalvi/code_workspace/tmp/csv_data/x_vel_5.csv', lin_vel, delimiter=',')
#savetxt('/home/asalvi/code_workspace/tmp/csv_data/z_ang_5.csv', z_ang, delimiter=',')
#savetxt('/home/asalvi/code_workspace/tmp/csv_data/x_pos_5.csv', x_pos, delimiter=',')
#savetxt('/home/asalvi/code_workspace/tmp/csv_data/y_pos_5.csv', y_pos, delimiter=',')


print('Done saving files')
