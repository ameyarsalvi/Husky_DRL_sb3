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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
import skops.io as sio
from skops.io import loads


import os

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/")

import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient



class HuskyCPEnvConeGP2(Env):
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
        self.observation_space = Box(low=0, high=255,shape=(96,320,1),dtype=np.uint8)
        

        # Initial decleration of variables

        # Some global initialization
        self.centroid_buffer =[]
        self.floor_x_goal = 25
        self.floor_y_goal = 15
        self.episode_length = 5000
        self.step_no = 0
        self.global_timesteps = 0
        self.z_ang = []
        self.Left = 0
        self.Right = 0
        self.V = 0
        self.omega = 0
        self.sigma1 = 0
        self.sigma2 = 0

        '''

        path = '/home/asalvi/Downloads/skops/'
        #import skops.io as sio
        from skops.io import load
        from skops.io import get_untrusted_types
        #with open(path + "beta1.pkl", "rb") as fp:   # Unpickling
        #    self.Beta1_grass = pickle.load(fp)

        #with open(path + "beta2.pkl", "rb") as fp:   # Unpickling
        #    self.Beta2_grass = pickle.load(fp)

        unknown_types = get_untrusted_types(file="beta1.skops")
        unknown_types = get_untrusted_types(file="beta2.skops")

        self.Beta1_grass = load("beta1.skops", trusted=unknown_types)
        self.Beta2_grass = load("beta2.skops", trusted=unknown_types)
        '''
        
        '''
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
        '''

    

    def step(self,action):
 
     
        # Take Action
        self.V = 0.25*action[0] + 0.75
        self.omega = 0.5*action[1]

        # Add GP predictions to V and Omega

        

        

        '''
        path = '/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/heat_map/all_final/'
        specifier = 'ten'
        self.log_actV .append(V)
        savetxt(path + specifier + '_actV3.csv', self.log_rel_vel_lin, delimiter=',')
        self.log_actV.append(omega)
        savetxt(path + specifier + '_actW2.csv', self.log_rel_vel_ang, delimiter=',')
        '''
        
        self.invKin()
    
        #Joint Velocities similar to how velocities are set on actual robot
        self.sim.setJointTargetVelocity(self.fl_w, self.Left)
        self.sim.setJointTargetVelocity(self.fr_w, self.Right)
        self.sim.setJointTargetVelocity(self.rl_w, self.Left)
        self.sim.setJointTargetVelocity(self.rr_w, self.Right)

        
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

        cropped_image = img[288:480, 0:640] # Crop image to only to relevant path data (Done heuristically)
        cropped_image = cv2.resize(cropped_image, (0,0), fx=0.5, fy=0.5)
        im_bw = cv2.threshold(cropped_image, 125, 200, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        im_bw = cv2.threshold(im_bw, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
        noisy_image = cv2.add(im_bw, noise)
        im_bw = np.frombuffer(noisy_image, dtype=np.uint8).reshape(96, 320, 1) # Reshape to required observation size
        im_bw_obs = ~im_bw

        
        crop_error = img[400:480, 192:448] # Crop image to only to relevant path data (Done heuristically)
        crop_error = cv2.threshold(crop_error, 225, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image

        #cv2.imshow("For Calc", crop_error)
        #cv2.imshow("obs", im_bw_obs)
        #cv2.waitKey(1)

        M = cv2.moments(crop_error) # calculate moments of binary image
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        #cX = 128


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
        if self.step_no == 0:
            self.img = im_bw_obs
        else:
            if self.step_no % 1 == 0:
                self.img = im_bw_obs
            else:
                pass
                
        
        
        self.state = np.array(self.img,dtype = np.uint8) #Just input image to CNN network

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

        # Linear velocity reward params

        linear_vel, angular_vel = self.sim.getVelocity(self.COM)
        sRb = self.sim.getObjectMatrix(self.COM,self.sim.handle_world)
        Rot = np.array([[sRb[0],sRb[1],sRb[2]],[sRb[4],sRb[5],sRb[6]],[sRb[8],sRb[9],sRb[10]]])
        vel_body = np.matmul(np.transpose(Rot),np.array([[linear_vel[0]],[linear_vel[1]],[linear_vel[2]]]))
        realized_vel = np.abs(-1*vel_body[2].item())
        track_vel = self.track_vel 
        err_vel = np.abs(track_vel - realized_vel)
        err_vel = np.clip(err_vel,0,0.5)
        norm_err_vel = (err_vel - 0)/(0.5) ##   << -------------- Normalized Linear Vel


        # Lane centering reward params
        err_track = np.abs(self.error)
        if err_track > 125:
            err_track = 125   
        else:
            pass
        norm_err_track = (err_track - 0)/125 ##   << -------------- Normalized Lane centering

        # Angular velocity reward params
        Gyro_Z = self.sim.getFloatSignal("myGyroData_angZ")
        #if Gyro_Z:
        ##    self.z_ang.append(Gyro_Z)
        err_effort = np.abs(Gyro_Z)     
        norm_err_eff = (err_effort)/0.6 ##   << -------------- Normalized angular velocity

        # Variance map
        self.get_map()
        var1_norm = (0.71 - self.sigma1**2)/0.71
        var2_norm = (1.1 - self.sigma2**2)/1.1

        # Total reward
        reward = (1 - norm_err_track)**2 + (1 - norm_err_vel)**2 +(1- norm_err_eff)**2 + (1-var1_norm)**2 + (1-var2_norm)**2
        reward = np.float64(reward)

        
        '''
        #Comment in/out depending on training or evaluation
        
        ### Data logging
        
        path = '/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf_all/'
        specifier = 'vp_' + str(int(100*self.track_vel))
        
        
        
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
        
        #spawn = self.sim.getObject('/Spawn1')
                    
        pose = self.sim.getObjectPose(spawn, self.sim.handle_world)
        self.sim.setObjectPose(self.COM, pose,self.sim.handle_world)

        ####### Sligthly reshuffle cones ############

        for x in range(100):

            cone1 = self.sim.getObject('/Cone[' + str(int(x+1)) + ']')
            cone1_pose = self.sim.getObjectPose(cone1, self.sim.handle_world)
            x_dist = 0.01*np.random.random(size=None)
            y_dist = 0.01*np.random.random(size=None)
            self.sim.setObjectPose(cone1, [cone1_pose[0]+x_dist, cone1_pose[1]+y_dist,cone1_pose[2], cone1_pose[3],cone1_pose[4],cone1_pose[5],cone1_pose[6]], self.sim.handle_world)


            cone2 = self.sim.getObject('/Cone2[' + str(int(x+1)) + ']')
            cone2_pose = self.sim.getObjectPose(cone2, self.sim.handle_world)
            x_dist = 0.01*np.random.random(size=None)
            y_dist = 0.01*np.random.random(size=None)
            self.sim.setObjectPose(cone2, [cone2_pose[0]+x_dist,cone2_pose[1]+y_dist,cone2_pose[2],cone2_pose[3],cone2_pose[4],cone2_pose[5],cone2_pose[6]], self.sim.handle_world)
            

        
    
        # IMAGE PROCESSING CODE (Only to send obseravation)
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensorHandle)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale

        # Current image
        cropped_image = img[288:480, 0:640] # Crop image to only to relevant path data (Done heuristically)
        cropped_image = cv2.resize(cropped_image, (0,0), fx=0.5, fy=0.5)
        im_bw = cv2.threshold(cropped_image, 125, 200, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        im_bw = cv2.threshold(im_bw, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
        noisy_image = cv2.add(im_bw, noise)
        im_bw = np.frombuffer(noisy_image, dtype=np.uint8).reshape(96, 320, 1) # Reshape to required observation size
        im_bw_obs = ~im_bw
        
        #Send observation for learning
        self.state = np.array(im_bw_obs,dtype = np.uint8) #Just input image to CNN network
        
        info = {}

        #self.sim.step()
        #clinet.step()

        return self.state, info
    
    def invKin(self):

        from skops.io import load
        from skops.io import get_untrusted_types
        #with open(path + "beta1.pkl", "rb") as fp:   # Unpickling
        #    self.Beta1_grass = pickle.load(fp)

        #with open(path + "beta2.pkl", "rb") as fp:   # Unpickling
        #    self.Beta2_grass = pickle.load(fp)

        unknown_types = get_untrusted_types(file="beta1.skops")
        unknown_types = get_untrusted_types(file="beta2.skops")

        Beta1_grass = load("beta1.skops", trusted=unknown_types)
        Beta2_grass = load("beta2.skops", trusted=unknown_types)

        V = self.V 
        omega = self.omega
        
        # No condition
        self.t_a = 0.7510
        self.t_b = 1.5818
        

        A = np.array([[self.t_a*0.0825,self.t_a*0.0825],[-0.1486/self.t_b,0.1486/self.t_b]])
        velocity = np.array([V,omega])
        #print(velocity)
        phi_dots = np.matmul(inv(A),velocity) #Inverse Kinematics
        phi_dots = phi_dots.astype(float)
        self.Left = phi_dots[0].item()
        self.Right = phi_dots[1].item()

        input = np.array([V,omega]).reshape((1,-1))

        mu1, sigma1 = Beta1_grass.predict(input, return_std = True)
        mu2, sigma2 = Beta2_grass.predict(input, return_std = True)

        #velocity = np.array([V - mu1, omega-mu2])
        #print(velocity)
        #phi_dots = np.matmul(inv(A),velocity) #Inverse Kinematics
        #phi_dots = phi_dots.astype(float)
        self.Left = self.Left + mu1.item()
        self.Right = self.Right + mu2.item()

        return self.Left, self.Right
    
    def get_map(self):

        from skops.io import load
        from skops.io import get_untrusted_types
        #with open(path + "beta1.pkl", "rb") as fp:   # Unpickling
        #    self.Beta1_grass = pickle.load(fp)

        #with open(path + "beta2.pkl", "rb") as fp:   # Unpickling
        #    self.Beta2_grass = pickle.load(fp)

        unknown_types = get_untrusted_types(file="beta1.skops")
        unknown_types = get_untrusted_types(file="beta2.skops")

        Beta1_grass = load("beta1.skops", trusted=unknown_types)
        Beta2_grass = load("beta2.skops", trusted=unknown_types)

        input = np.array([self.V,self.omega]).reshape((1,-1))

        mu1, sigma1 = Beta1_grass.predict(input, return_std = True)
        self.sigma1 = sigma1.item()
        mu2, sigma2 = Beta2_grass.predict(input, return_std = True)
        self.sigma2 = sigma2.item()
        
        return self.sigma1, self.sigma2



    

    



'''
Environment Validation code

#Comment out three lines to validated environment

from stable_baselines3.common.env_checker import check_env
env = HuskyCPEnvCone()
check_env(env)
'''
