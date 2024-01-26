
import numpy as np
from numpy import linalg
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box 
import random 
import torch
import cv2
from numpy.linalg import inv

import os

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/")

import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient



class HuskyCPEnv(Env):
    def __init__(self,port):

        #Initializing socket connection
        client = RemoteAPIClient('localhost',port)
        #self.sim = client.getObject('sim')
        self.sim = client.require('sim')
        self.sim.setStepping(True)
        self.sim.startSimulation()
        #while self.sim.getSimulationState() == self.sim.simulation_stopped:
        #    time.sleep(5)

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
        '''
        #These are some declerations from Go to goal

        # Linear and angular velocities
        linear_vel = self.sim.getObjectVelocity(self.COM,self.sim.handle_world)
        self.lin_velocity = linear_vel[0]
        self.ang_velocity = self.sim.getFloatSignal("myGyroData_angZ")
        # Robot position
        bot_position = self.sim.getObjectPosition(self.COM,self.sim.handle_world)
        self.bot_x_pos = bot_position[0]
        self.bot_y_pos = bot_position[1]
        # Get Origig (robot spawns/ resets here)
        floor_position = self.sim.getObjectPosition(self.floor_cen,self.sim.handle_world)
        self.floor_x_pos = floor_position[0]
        self.floor_y_pos = floor_position[1]
        # Eucledian Distance of the robot from origin
        x_norm_origin = linalg.norm([self.floor_x_pos,self.bot_x_pos])
        y_norm_origin = linalg.norm([self.floor_y_pos,self.bot_y_pos])
        # Eucledian Distance of robot from goal
        x_norm_goal = linalg.norm([self.floor_x_goal,self.bot_x_pos])
        y_norm_goal = linalg.norm([self.floor_y_goal,self.bot_y_pos])
        self.goal_dist = np.sum([x_norm_goal,y_norm_goal])
        self.away = np.sum([x_norm_origin,y_norm_origin])
        '''

    def step(self,action):

        
     
        # Take Action
        ## Control map: (Please keep this code to know what control matrix for inverse kinematics)
        # x_dot = [0.081675 0.081675; -0.1081 -0.1081] phi_dot
        # A_low = np.array([[0.0701,0.0701],[-0.1351,0.1351]])
        # A_med = np.array([[0.0743,0.0743],[-0.1025,0.1025]])
        # A_high = np.array([[0.0825,0.0825],[-0.1122,0.1122]])
        
        #A = np.array([[0.0825,0.0825],[-0.1122,0.1122]])
        A = np.array([[0.0701,0.0701],[-0.1351,0.1351]])
        #V = 0.15*action[0] + 0.65 #V range : [0.5 0.8] 
        V = 0.45*action[0] + 0.55 # >> Constrain to [0.6 0.7] >> Complete space 0 -> 1
        omega = 0.5*action[1] #Omega range : [-0.5 0.5]
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
        #cv2.imshow("Image Now", crop_error)
        #cv2.waitKey(1)

        #Calcuation of centroid for lane centering
        M = cv2.moments(crop_error) # calculate moments of binary image
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
    
        
        #This code is to plot centroid on the processed image and display it
        # put text and highlight the center
        #cv2.circle(crop_error, (cX, cY), 5, (255, 255, 255), -1)
        #cv2.putText(crop_error, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        #resize_img = cv2.resize(cropped_image  , (10 , 5))
    
        # display the image
        #cv2.imshow("Image", crop_error)
        #cv2.waitKey(1)
        


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
        self.state = np.array(im_bw,dtype = np.uint8) #Just input image to CNN network

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
        track_vel = 1
        err_vel = np.abs(track_vel - realized_vel)
        err_track = np.abs(self.error)
        if err_track > 50:
            err_track = 50   
        else:
            pass        
        norm_err_vel = (err_vel - 0)/(0.9)
        norm_err_track = (err_track -0)/50
        norm_err_step = (self.step_no -0)/5000
        #reward = 2*(norm_err_step/(norm_err_track+0.001)) - 0*norm_err_vel
        reward = (1 - norm_err_track)**2 + (1 - norm_err_vel)**2
        #reward = np.float64(reward)
        #reward = 1 - norm_err_track
        reward = np.float64(reward)


        # Check for reset conditions
        # Removing episode length termination from reset condition
        if self.episode_length ==0 or err_track>50 or reset == 1:
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
        super().reset(seed=seed)

        # Reset initialization variables
        self.centroid_buffer = []
        self.episode_length = 5000
        self.step_no = 0

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
