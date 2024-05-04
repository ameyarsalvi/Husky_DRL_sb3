
import numpy as np
from numpy import linalg
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box 
import random 
import torch
import cv2
from numpy.linalg import inv
from numpy import savetxt
import pickle
from matplotlib.animation import FuncAnimation


import os

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/")

import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient



class HuskyCPEnv(Env):
    def __init__(self,port,seed,track_vel):

        #Initializing socket connection
        client = RemoteAPIClient('localhost',port)
        #self.sim = client.getObject('sim')
        self.sim = client.require('sim')
        self.sim.setStepping(True)
        self.sim.startSimulation()
        #while self.sim.getSimulationState() == self.sim.simulation_stopped:
        #    time.sleep(5)

        self.seed = seed
        self.track_vel = track_vel

        self.flw_vel = 0

        #Get object handles from CoppeliaSim handles
        self.visionSensorHandle = self.sim.getObject('/Vision_sensor')
        self.fl_w = self.sim.getObject('/flw')
        self.fr_w = self.sim.getObject('/frw')
        self.rr_w = self.sim.getObject('/rrw')
        self.rl_w = self.sim.getObject('/rlw')
        self.IMU = self.sim.getObject('/Accelerometer_forceSensor')
        self.COM = self.sim.getObject('/Husky')
        self.floor_cen = self.sim.getObject('/Floor')
        self.t_a = 0
        self.t_b = 0

        # Limits and definitions on Observation and action space
        
        #Action space def : [Left wheel velocity (rad/s), Right wheel velocity (rad/s)]
        self.action_space = Box(low=np.array([[-1],[-1]]), high=np.array([[1],[1]]),dtype=np.float32)

        # Observation shape definition : [Image pixels of size 64x256x1]
        #self.observation_space = Box(low=np.array([[-1000],[-1000],[-1000],[-1000]]), high=np.array([[1000],[1000],[1000],[1000]]),dtype=np.float32)
        self.observation_space = Box(low=0, high=255,shape=(192,256,1),dtype=np.uint8)
        

        # Initial decleration of variables

        # Some global initialization
        self.centroid_buffer =[]
        self.floor_x_goal = 25
        self.floor_y_goal = 15
        self.episode_length = 5000
        self.step_no = 0
        self.global_timesteps = 0
        self.z_ang = []
        #log variables
        self.log_err_feat = []
        self.log_err_vel = []
        self.log_err_feat_norm = []
        self.log_err_vel_norm = []
        self.log_rel_vel_lin = []
        self.log_rel_vel_ang = []
        self.log_actV = []
        self.log_actW = []
        self.log_err_omega = []
        self.log_err_omega_norm = []


    def step(self,action):

        
     
        # Take Action
        ## Control map: (Please keep this code to know what control matrix for inverse kinematics)
        # x_dot = [0.081675 0.081675; -0.1081 -0.1081] phi_dot
        # A_low = np.array([[0.0701,0.0701],[-0.1351,0.1351]])
        # A_med = np.array([[0.0743,0.0743],[-0.1025,0.1025]])
        # A_high = np.array([[0.0825,0.0825],[-0.1122,0.1122]])
     
        
        #V = 0.25*action[0] + 0.75 
        #print(action[0].item())
        V = 0.25*action[0] + 0.75
        omega = 0.6*action[1]

        

        '''
        path = '/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/heat_map/all_final/'
        specifier = 'ten'
        self.log_actV .append(V)
        savetxt(path + specifier + '_actV3.csv', self.log_rel_vel_lin, delimiter=',')
        self.log_actV.append(omega)
        savetxt(path + specifier + '_actW2.csv', self.log_rel_vel_ang, delimiter=',')
        '''
        
        '''
        # Ten condition
        condtion = np.abs(omega)
        if condtion > 0.55:
            self.t_a = 0.5870
            self.t_b = 2.5223
        elif 0.5> condtion <=0.55:
            self.t_a = 0.6
            self.t_b = 2.3782
        elif 0.45> condtion <=0.5:
            self.t_a = 0.6356
            self.t_b = 2.1620
        elif 0.4> condtion <=0.45:
            self.t_a = 0.6713
            self.t_b = 1.9818
        elif 0.35> condtion <=0.4:
            self.t_a = 0.7070
            self.t_b = 1.7656
        elif 0.3> condtion <=0.35:
            self.t_a = 0.7485
            self.t_b = 1.5854
        elif 0.25> condtion <=0.3:
            self.t_a = 0.7975
            self.t_b = 1.2972
        elif 0.2> condtion <=0.25:
            self.t_a = 0.8597
            self.t_b = 1.0089
        elif 0.15> condtion <=0.2:
            self.t_a = 0.9214
            self.t_b = 0.7207
        elif 0.0 > condtion <= 0.15:
            self.t_a = 0.9821
            self.t_b = 0.3963
        '''
        
        '''
          # Five condition
        condtion = np.abs(omega)
        if condtion > 0.4:
            self.t_a = 0.5935
            self.t_b = 2.4502
        elif 0.3> condtion <=0.4:
            self.t_a = 0.6355
            self.t_b = 2.0719
        elif 0.2> condtion <=0.3:
            self.t_a = 0.7278
            self.t_b = 1.6755
        elif 0.1> condtion <=0.2:
            self.t_a = 0.8286
            self.t_b = 1.1531
        elif 0.0 > condtion <= 0.1:
            self.t_a = 0.9517
            self.t_b = 0.5585
        '''
        
        
        '''
          # Three condition
        condtion = np.abs(omega)
        if condtion > 0.4:
            self.t_a = 0.6076
            self.t_b = 2.3541
        elif 0.2 > condtion <=0.4:
            self.t_a = 0.7311
            self.t_b = 1.6575
        elif 0.0 > condtion <= 0.2:
            self.t_a = 0.9210
            self.t_b = 0.7086
        
        '''

        
        # No condition
        self.t_a = 0.7510
        self.t_b = 1.5818
        
        '''
        # DR
        self.t_a = 0.7510 + np.random.normal(0, 0.035, 1).item()
        self.t_b = 1.5818 + np.random.normal(0, 0.085, 1).item()
        '''


        A = np.array([[self.t_a*0.0825,self.t_a*0.0825],[-0.1486/self.t_b,0.1486/self.t_b]])
        velocity = np.array([V,omega])
        phi_dots = np.matmul(inv(A),velocity) #Inverse Kinematics
        phi_dots = phi_dots.astype(float)
        Left = phi_dots[0].item()
        Right = phi_dots[1].item()
    
        #Joint Velocities similar to how velocities are set on actual robot
        self.sim.setJointTargetVelocity(self.fl_w, Left)
        self.sim.setJointTargetVelocity(self.fr_w, Right)
        self.sim.setJointTargetVelocity(self.rl_w, Left)
        self.sim.setJointTargetVelocity(self.rr_w, Right)

        
         # Simulate step (CoppeliaSim Command) (Progress on the taken action)
        self.sim.step()
     
        # Get simulation data
        # IMAGE PROCESSING CODE 
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensorHandle)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale

        # Current image
        cropped_image = img[288:480, 192:448] # Crop image to only to relevant path data (Done heuristically)
        crop_error = img[400:480, 192:448] # Crop image to only to relevant path data (Done heuristically)
        im_bw = cv2.threshold(cropped_image, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
        noisy_image = cv2.add(im_bw, noise)
        im_bw = np.frombuffer(noisy_image, dtype=np.uint8).reshape(192, 256, 1) # Reshape to required observation size
        im_bw_obs = ~im_bw
        #cv2.imshow("For Calc", crop_error)
        #cv2.imshow("obs", im_bw)
        #cv2.waitKey(1)

        #Calcuation of centroid for lane centering
        crop_error = ~crop_error
        M = cv2.moments(crop_error) # calculate moments of binary image
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
    


        # Centering Error :
        self.error = 128 - cX #Lane centering error that can be used in reward function
        #print(self.error)
        self.centroid_buffer.append(self.error) #This maintains a size 10 buffer which resets if all elements are -2 indicating the robot has strayed
        res = self.centroid_buffer[-10:]
        #if np.sum(res) == -20:
        #    reset = 1
        #else:
        #    reset = 0
        # Updating reset condition to sum of pixels
        #print(self.error)
        #print(np.sum(im_bw).item())

        
        if np.sum(im_bw).item() == 12533760:
            reset = 1
        else:
            reset = 0
        

        #Send observation for learning
        self.state = np.array(im_bw_obs,dtype = np.uint8) #Just input image to CNN network

        #Display what is being sent as an observation for training :
        #cv2.imshow("Image", im_bw)
        #cv2.waitKey(1)


        # Calculate Reward
        '''
        This reward has three terms :
        1. Increase episode count (This just encourages to stay on the track)
        2. Increase linear velocity (This encourages to move so that the robot doesn't learn a trivial action)
        3. Reduce center tracing error (This encourages smoothening)
        '''
        #rew_atr = np.array([self.step_no**2,V.item(),self.error]) # Reward Attributes 
        #rew_wgt = np.array([10,10,0])
        #reward = np.float64(np.dot(rew_atr,rew_wgt))

        # No risk
        #inc_wt = 4*self.global_timesteps*1e-6 
        # Get robot realized linear velocity
        linear_vel, angular_vel = self.sim.getVelocity(self.COM)
        sRb = self.sim.getObjectMatrix(self.COM,self.sim.handle_world)
        Rot = np.array([[sRb[0],sRb[1],sRb[2]],[sRb[4],sRb[5],sRb[6]],[sRb[8],sRb[9],sRb[10]]])
        vel_body = np.matmul(np.transpose(Rot),np.array([[linear_vel[0]],[linear_vel[1]],[linear_vel[2]]]))
        realized_vel = np.abs(-1*vel_body[2].item())
        self.log_rel_vel_lin.append(realized_vel)
        track_vel = self.track_vel ## << -- study update
        err_vel = np.abs(track_vel - realized_vel)
        err_vel = np.clip(err_vel,0,0.5)
        err_track = np.abs(self.error)
        if err_track > 125:
            err_track = 125   
        else:
            pass
        err_effort = np.abs(omega)     
        norm_err_vel = (err_vel - 0)/(0.5)
        norm_err_track = (err_track - 0)/125
        norm_err_step = (self.step_no -0)/5000
        norm_err_eff = (err_effort)/0.6

        Gyro_Z = self.sim.getFloatSignal("myGyroData_angZ")
        if Gyro_Z:
            self.z_ang.append(Gyro_Z)

        #Pretraining 
        #omega_des = -0.05*self.error
        #act_error = omega_des - omega
        #norm_act_error = (act_error + 1 )/2
        #reward = (1 - norm_act_error)**2

        reward = (1 - norm_err_track)**2 + (1 - norm_err_vel)**2 +(1- norm_err_eff)**2
        #reward = np.float64(reward)
        #reward = 1 - norm_err_track
        reward = np.float64(reward)

        
        #self.log_err_feat = err_track.tolist()
        
        #Comment in/out depending on training or evaluation
        
        ### Data logging
        
        path = '/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf_all/'
        specifier = 'vp_' + str(int(100*self.track_vel))

        '''
        self.log_err_feat.append(err_track)
        savetxt(path + specifier + '_err_feat.csv', self.log_err_feat, delimiter=',')
        self.log_err_vel.append(err_vel)
        savetxt(path + specifier + '_err_vel.csv', self.log_err_vel, delimiter=',')
        self.log_err_feat_norm.append(norm_err_track)
        savetxt(path + specifier + '_err_feat_norm.csv', self.log_err_feat_norm, delimiter=',')
        self.log_err_vel_norm.append(norm_err_vel)
        savetxt(path + specifier + '_err_vel_norm.csv', self.log_err_vel_norm, delimiter=',')

        self.log_rel_vel_lin.append(realized_vel)
        savetxt(path + specifier + '_rel_vel_lin.csv', self.log_rel_vel_lin, delimiter=',')
        self.log_rel_vel_ang.append(Gyro_Z)
        savetxt(path + specifier + '_rel_vel_ang.csv', self.log_rel_vel_ang, delimiter=',')
        '''
        
        '''
        
        self.log_err_feat.append(err_track)
        with open(path + specifier + "_err_feat", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_feat, fp)
        
        self.log_err_feat_norm.append(norm_err_track)
        with open(path + specifier + "_err_feat_norm", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_feat_norm, fp)

        self.log_err_vel.append(err_vel)
        with open(path + specifier + "_err_vel", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_vel, fp)

        self.log_err_vel_norm.append(norm_err_vel)
        with open(path + specifier + "_err_vel_norm", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_vel_norm, fp)

        self.log_err_omega.append(err_effort)
        with open(path + specifier + "_err_omega", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_vel, fp)

        self.log_err_omega_norm.append(norm_err_eff)
        with open(path + specifier + "_err_omega_norm", "wb") as fp:   #Pickling
            pickle.dump(self.log_err_vel_norm, fp)
        
        self.log_rel_vel_lin.append(realized_vel)
        with open(path + specifier + "_rel_vel_lin", "wb") as fp:   #Pickling
            pickle.dump(self.log_rel_vel_lin, fp)
        
        self.log_rel_vel_ang.append(Gyro_Z)
        with open(path + specifier + "_rel_vel_ang", "wb") as fp:   #Pickling
            pickle.dump(self.log_rel_vel_ang, fp)

        self.log_actV .append(V)
        with open(path + specifier + "_actV", "wb") as fp:   #Pickling
            pickle.dump(self.log_actV, fp)
        
        self.log_actW.append(omega)
        with open(path + specifier + "_actW", "wb") as fp:   #Pickling
            pickle.dump(self.log_actW, fp)
        
        '''
        

        # Check for reset conditions
        # Removing episode length termination from reset condition
        if self.episode_length ==0 or np.abs(self.error)>125 or reset == 1:
            done = True
        else:
            done = False


        # Update Global variables
        self.episode_length -= 1 #Update episode step counter
        self.step_no += 1 
        self.global_timesteps +=1   


        info ={}

        return self.state, reward, done, False, info

    def render(self):
        pass

    def reset(self, seed=None):
        super().reset(seed=self.seed)

        # Reset initialization variables
        self.centroid_buffer = []
        self.episode_length = 5000
        self.step_no = 0
        self.z_ang = []

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        self.sim.setStepping(True)
        self.sim.startSimulation() 

        # Randomize spawning location so that learning is a bit more generalized
        # Three objects kept at different locations
        rand_spawn = np.random.randint(1, 9, 1, dtype=int)
        if rand_spawn == 1:
            spawn = self.sim.getObject('/Spawn1')   
        elif rand_spawn == 2:
            spawn = self.sim.getObject('/Spawn2')
        elif rand_spawn == 3:
            spawn = self.sim.getObject('/Spawn3')
        elif rand_spawn == 4:
            spawn = self.sim.getObject('/Spawn4')
        elif rand_spawn == 5:
            spawn = self.sim.getObject('/Spawn5')
        elif rand_spawn == 6:
            spawn = self.sim.getObject('/Spawn6')
        elif rand_spawn == 7:
            spawn = self.sim.getObject('/Spawn7')
        else :
            spawn = self.sim.getObject('/Spawn8') 
                    
        pose = self.sim.getObjectPose(spawn, self.sim.handle_world)
        self.sim.setObjectPose(self.COM, pose,self.sim.handle_world)
        
    
        # IMAGE PROCESSING CODE (Only to send obseravation)
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensorHandle)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale

        # Current image
        cropped_image = img[288:480, 192:448] # Crop image to only to relevant path data (Done heuristically)
        im_bw = cv2.threshold(cropped_image, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
        noisy_image = cv2.add(im_bw, noise)
        im_bw = np.frombuffer(noisy_image, dtype=np.uint8).reshape(192, 256, 1) # Reshape to required observation size
        im_bw = np.frombuffer(im_bw, dtype=np.uint8).reshape(192, 256, 1) # Reshape to required observation size
        
        #Send observation for learning
        self.state = np.array(im_bw,dtype = np.uint8) #Just input image to CNN network
        
        info = {}

        #self.sim.step()
        #clinet.step()

        return self.state, info
        


'''
Environment Validation code

#Comment out three lines to validated environment

#from stable_baselines3.common.env_checker import check_env
#env = HuskyCPEnv()
#check_env(env)

'''


