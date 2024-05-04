import numpy as np
from numpy import linalg
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box 
from gymnasium.spaces import Dict
import random 
import torch
import cv2
from numpy.linalg import inv
from numpy import savetxt
import pickle
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import os

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/")

import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient



class HuskyCPEnvPathFren(Env):
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
        self.throttle = 1

        self.flw_vel = 0
        self.cam_err_old = 0

        #Get object handles from CoppeliaSim handles
        self.visionSensorHandle = self.sim.getObject('/Vision_sensor')
        self.fl_w = self.sim.getObject('/flw')
        self.fr_w = self.sim.getObject('/frw')
        self.rr_w = self.sim.getObject('/rrw')
        self.rl_w = self.sim.getObject('/rlw')
        self.IMU = self.sim.getObject('/Accelerometer_forceSensor')
        self.COM = self.sim.getObject('/Husky')
        self.floor_cen = self.sim.getObject('/Floor')
        self.BodyFOR = self.sim.getObject('/FORBody')
        self.HuskyPos = self.sim.getObject('/FORBody/Husky/ReferenceFrame')
        self.CameraJoint = self.sim.getObject('/FORBody/Husky/Camera_joint')


        # Limits and definitions on Observation and action space
        
        #Action space def : [Left wheel velocity (rad/s), Right wheel velocity (rad/s)]
        #self.action_space = Box(low=np.array([[-1],[-1],[-1]]), high=np.array([[1],[1],[1]]),dtype=np.float32)
        self.action_space = Box(low=np.array([[-1],[-1]]), high=np.array([[1],[1]]),dtype=np.float32)
        #self.action_space = Box(low=np.array([[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1]]), high=np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]),dtype=np.float32)

        # Observation shape definition : [Image pixels of size 64x256x1]
        #self.observation_space = Box(low=np.array([[-1000],[-1000],[-1000],[-1000]]), high=np.array([[1000],[1000],[1000],[1000]]),dtype=np.float32)
        self.observation_space = Box(low=0, high=255,shape=(105,320,1),dtype=np.uint8)

        '''
        self.observation_space = Dict(
            {
                "image" : Box(low=0, high=255,shape=(1,105,320),dtype=np.uint8),
                "camera_angle" : Box(low=0, high=1, shape=(1,), dtype=np.uint8)
            }
        )
        '''

        # Initial decleration of variables

        # Some global initialization
        # Reset initializations
        '''
        self.centroid_buffer =[]
        self.episode_length = 5000
        self.step_no = 0
        self.z_ang = []
        self.current_pose =[]
        self.prev_pose = []
        self.ArcLen = []
        self.path_track_err = []
        self.intg_pth_err = []
        self.curretn_orn = []
        self.pose_track_err = []
        self.path_selector = []
        self.throttle = 1 

        self.error = 0
        self.camActSet = []
        self.camActRel = []
        self.camErr = []
        self.camErrPrv = 0
        '''
        self.obs = []
        self.img_div = 1




        # Paths
        from numpy import genfromtxt
        path_loc = '/home/asalvi/code_workspace/Husky_CS_SB3/Paths/InfPath2/'
        self.path1 = genfromtxt(path_loc + 'ArcPath1.csv', delimiter=',')
        self.path2 = genfromtxt(path_loc + 'ArcPath2.csv', delimiter=',')
        self.path3 = genfromtxt(path_loc + 'ArcPath3.csv', delimiter=',')
        self.path4 = genfromtxt(path_loc + 'ArcPath4.csv', delimiter=',')
        self.path5 = genfromtxt(path_loc + 'ArcPath5.csv', delimiter=',')
        
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
        #V = 0.45*action[0] + 0.5 # 
        #omega = 0.75*action[1]
        
        #Left = 6*(action[0].item())
        #Right = 6*(action[0].item())
        #cam_ang = 45*action[2]
        

        '''
        path = '/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/heat_map/all_final/'
        specifier = 'ten'
        self.log_actV .append(V)
        savetxt(path + specifier + '_actV3.csv', self.log_rel_vel_lin, delimiter=',')
        self.log_actV.append(omega)
        savetxt(path + specifier + '_actW2.csv', self.log_rel_vel_ang, delimiter=',')
        '''

        #Update actions setpoints
        '''
        if self.step_no == 0:
            self.path_speed = 0.45*action[0] + 0.55
            self.path_curve = action[1]*7.5
            
        else:
            if self.step_no % self.throttle == 0:
                self.path_speed = 0.45*action[0] + 0.55
                self.path_curve = action[1]*7.5
            else:
                pass
        '''       

        
        
        self.GenControl(action)
        
    
        #Joint Velocities similar to how velocities are set on actual robot
        self.sim.setJointTargetVelocity(self.fl_w, self.Left)
        self.sim.setJointTargetVelocity(self.fr_w, self.Right)
        self.sim.setJointTargetVelocity(self.rl_w, self.Left)
        self.sim.setJointTargetVelocity(self.rr_w, self.Right)
        # Adjust Camera
        #cam_angle = 1*1*self.error # In degrees
        #cam_angle = np.clip(cam_angle,-45,45)

        #self.sim.setJointPosition(self.CameraJoint, self.camActRel)
        #self.sim.setJointPosition(self.CameraJoint, cam_ang.item()*math.pi/180)

        
         # Simulate step (CoppeliaSim Command) (Progress on the taken action)
        self.sim.step()
     
        # Get simulation data
        # IMAGE PROCESSING CODE
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensorHandle)
        self.img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        self.img_preprocess()

        
        #Send observation for learning
        '''
        if self.step_no == 0:
            self.img_obs = self.obs
        else:
            if self.step_no % self.throttle == 0:
                self.img_obs = self.obs
            else:
                pass
        '''        
        img_obs = self.obs
        img_obs = np.divide(img_obs,self.img_div)
        
        self.state = np.array(img_obs,dtype = np.uint8) #Just input image to CNN network
        #self.state = np.array(img_obs,0.5)
        
        #self.state['image'] = np.array(img_obs,dtype = np.uint8)
        #self.state['camera_angle'] = np.array(0.5,dtype = np.uint8)

        #Display what is being sent as an observation for training :
        #cv2.imshow("Image", im_bw)
        #cv2.waitKey(1)

        self.getReward()
        reward = self.rew
       

        #self.update_line()

        #self.Logger()
        
        

        # Check for reset conditions
        # Removing episode length termination from reset condition
        #if self.episode_length ==0 or np.abs(self.error)>125 or reset == 1:
        if self.episode_length ==0 or np.abs(self.path_track_err)>1 or np.abs(self.pose_track_err)>85 :
            done = True
        else:
            done = False


        # Update Global variables
        self.episode_length -= 1 #Update episode step counter
        self.step_no += 1  

        info ={}

        return self.state, reward, done, False, info

    def render(self):
        pass

    def reset(self, seed=None):
        super().reset(seed=self.seed)

        # Reset initialization variables
        self.episode_length = 5000
        self.step_no = 0
        self.z_ang = []
        self.ArcLen = 0
        self.obs = []
        self.camActPrev = 0
        self.camErrPrev = 0
        self.cam_err_old = 0


        #self.intg_pth_err = []
        #self.track_vel = 0.6 + 0.4*np.random.random(size=None)

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        self.sim.setStepping(True)
        self.sim.startSimulation() 

        # Randomize spawning location so that learning is a bit more generalized
        # Three objects kept at different locations
        
        rand_spawn = np.random.randint(1, 6, 1, dtype=int)
        #rand_spawn = 5
        if rand_spawn == 1:
            # Create a rotation object from Euler angles specifying axes of rotation
            rot = Rotation.from_euler('xyz', [0, 0,  self.path1[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path1[0,0],self.path1[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 1

        elif rand_spawn == 2:
            rot = Rotation.from_euler('xyz', [0, 0, self.path2[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path2[0,0],self.path2[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 2

        elif rand_spawn == 3:
            rot = Rotation.from_euler('xyz', [0, 0, self.path3[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path3[0,0],self.path3[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 3

        elif rand_spawn == 4:
            rot = Rotation.from_euler('xyz', [0, 0, self.path4[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path4[0,0],self.path4[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 4

        elif rand_spawn == 5:
            rot = Rotation.from_euler('xyz', [0, 0, self.path5[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            pose = [self.path5[0,0],self.path5[0,1],0.325,rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]
            self.path_selector = 5
    
        
        #spawn = self.sim.getObject('/Spawn1')
                    
        #pose = self.sim.getObjectPose(spawn, self.sim.handle_world)
        self.sim.setObjectPose(self.BodyFOR, pose,self.sim.handle_world)
        self.prev_pose = self.sim.getObjectPose(self.HuskyPos, self.sim.handle_world)

        ####### Sligthly reshuffle cones ############
        '''
        for x in range(100):

            cone1 = self.sim.getObject('/Cone[' + str(int(x+1)) + ']')
            cone1_pose = self.sim.getObjectPose(cone1, self.sim.handle_world)
            x_dist = 0.25*np.random.random(size=None)
            y_dist = 0.25*np.random.random(size=None)
            self.sim.setObjectPose(cone1, [cone1_pose[0]+x_dist, cone1_pose[1]+y_dist,cone1_pose[2], cone1_pose[3],cone1_pose[4],cone1_pose[5],cone1_pose[6]], self.sim.handle_world)


            cone2 = self.sim.getObject('/Cone2[' + str(int(x+1)) + ']')
            cone2_pose = self.sim.getObjectPose(cone2, self.sim.handle_world)
            x_dist = 0.25*np.random.random(size=None)
            y_dist = 0.25*np.random.random(size=None)
            self.sim.setObjectPose(cone2, [cone2_pose[0]+x_dist,cone2_pose[1]+y_dist,cone2_pose[2],cone2_pose[3],cone2_pose[4],cone2_pose[5],cone2_pose[6]], self.sim.handle_world)
        '''
        
    
        # IMAGE PROCESSING CODE (Only to send obseravation)
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensorHandle)
        self.img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        self.img_preprocess()
        img_obs = self.obs
        img_obs = np.divide(img_obs,2)
        

        #self.state = self.observation_space.sample()
        #Send observation for learning
        #self.state['image'] = np.array(img_obs,dtype = np.uint8)
        #self.state['camera_angle'] = np.array(0.5,dtype = np.uint8)
        self.state = np.array(img_obs,dtype = np.uint8) #Just input image to CNN network
        
        info = {}

        #self.sim.step()
        #clinet.step()

        return self.state, info
    
    def arc_length(self):
        
        self.current_pose = self.sim.getObjectPose(self.HuskyPos, self.sim.handle_world)
        #self.curretn_orn = np.array([self.current_pose[3],self.current_pose[4],self.current_pose[5],self.current_pose[6]]) #
        dt_arc = np.sqrt(np.square(self.current_pose[0]-self.prev_pose[0]) + np.square(self.current_pose[1]-self.prev_pose[1]))
        self.ArcLen = self.ArcLen + dt_arc

        if self.path_selector ==1:
            icp = min(enumerate(self.path1[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            des_poseX = self.path1[icp[0],0]
            des_poseY = self.path1[icp[0],1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path1[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            #des_poseQ = np.array([self.path1[icp[0],6],self.path1[icp[0],5],self.path1[icp[0],4],self.path1[icp[0],3]])
        elif self.path_selector == 2:
            icp = min(enumerate(self.path2[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            des_poseX = self.path2[icp[0],0]
            des_poseY = self.path2[icp[0],1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path2[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            #des_poseQ = np.array([self.path2[icp[0],6],self.path2[icp[0],5],self.path2[icp[0],4],self.path2[icp[0],3]])
        elif self.path_selector == 3:
            icp = min(enumerate(self.path3[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            des_poseX = self.path3[icp[0],0]
            des_poseY = self.path3[icp[0],1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path3[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ =np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            #des_poseQ = np.array([self.path3[icp[0],6],self.path3[icp[0],5],self.path3[icp[0],4],self.path3[icp[0],3]])
        elif self.path_selector == 4:
            icp = min(enumerate(self.path4[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            des_poseX = self.path4[icp[0],0]
            des_poseY = self.path4[icp[0],1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path4[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
            #des_poseQ = np.array([self.path4[icp[0],6],self.path4[icp[0],5],self.path4[icp[0],4],self.path4[icp[0],3]])
        elif self.path_selector == 5:
            icp = min(enumerate(self.path5[:,9]), key=lambda x: abs(x[1]-self.ArcLen))
            des_poseX = self.path5[icp[0],0]
            des_poseY = self.path5[icp[0],1]
            rot = Rotation.from_euler('xyz', [0, 0, self.path5[0,2]], degrees=False)
            rot_quat = rot.as_quat()
            des_poseQ = np.array([[rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3]]])
        #    des_poseQ = np.array([self.path5[icp[0],6],self.path5[icp[0],5],self.path5[icp[0],4],self.path5[icp[0],3]])



        self.path_track_err = np.sqrt(np.square(self.current_pose[0]-des_poseX) + np.square(self.current_pose[1]-des_poseY)) 
        #self.pose_track_err = np.arccos(self.curretn_orn@des_poseQ) #
        currentQ = np.array([[self.current_pose[3],self.current_pose[4],self.current_pose[5],self.current_pose[6]]])
        self.pose_track_err = 1- np.dot(des_poseQ,np.transpose(currentQ))
        #self.intg_pth_err.append(self.path_track_err)

        self.prev_pose = self.current_pose

    #def update_line(self):
    #    self.hl.set_xdata(np.append(self.hl.get_xdata(), self.step_no))
    #    self.hl.set_ydata(np.append(self.get_ydata(), self.realized_vel))
    #    plt.draw()
        

    def GenControl(self,action):

        ###### Wheel Velocities

        self.V_lin = 0.45*action[0] + 0.55
        self.Omg_ang = 0.75*action[0] 
        #camActSet = self.camAngle

        # No condition
        self.t_a = 0.7510
        self.t_b = 1.5818
        
        
        
        A = np.array([[self.t_a*0.0825,self.t_a*0.0825],[-0.1486/self.t_b,0.1486/self.t_b]])
        velocity = np.array([self.V_lin,self.Omg_ang])
        phi_dots = np.matmul(inv(A),velocity) #Inverse Kinematics
        phi_dots = phi_dots.astype(float)
        self.Left = phi_dots[0].item()
        self.Right = phi_dots[1].item()

        #### Camera Angle

        CamPosition = self.sim.getJointPosition(self.CameraJoint)
        #print(CamPosition)
        cam_error = (160-self.cX)
        CamFeed = 0.01*cam_error + 0.005*(np.abs(cam_error-self.cam_err_old))
        camSet = np.clip(CamFeed*(3.14/180),-3.14/2,3.14/2)
        self.cam_err_old = cam_error
        self.sim.setJointPosition(self.CameraJoint, CamPosition +  camSet.item())
        
        self.img_div = camSet.item()/(0.5*3.14)
        self.img_div +=2

    def img_preprocess(self):

        
        # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
        # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
        # and color format is RGB triplets, whereas OpenCV uses BGR:
        img0 = cv2.flip(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale
        cropped_image = img0[270:480, 0:640] # Crop image to only to relevant path data (Done heuristically)
        cropped_image = cv2.resize(cropped_image, (0,0), fx=0.5, fy=0.5)
        im_bw = cv2.threshold(cropped_image, 125, 200, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        im_bw = cv2.threshold(im_bw, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
        noisy_image = cv2.add(im_bw, noise)
        im_bw = np.frombuffer(noisy_image, dtype=np.uint8).reshape(105, 320,1) # Reshape to required observation size
        self.obs = ~im_bw

        
        cropped_image = img0[270:300, 0:640] # Crop image to only to relevant path data (Done heuristically)
        cropped_image = cv2.resize(cropped_image, (0,0), fx=0.5, fy=0.5)
        im_bw = cv2.threshold(cropped_image, 125, 200, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        im_bw = cv2.threshold(im_bw, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
        noisy_image = cv2.add(im_bw, noise)
        k = ~noisy_image
        M = cv2.moments(k)
 
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            self.cX = int(M["m10"] / M["m00"])
            self.cY = int(M["m01"] / M["m00"])
        else:
            self.cX, self.cY = 0, 0
    '''
    def lane_center(self):
        img0 = cv2.flip(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale
        crop_error = img0[400:480, 192:448] # Crop image to only to relevant path data (Done heuristically)
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
    '''

    def getReward(self):

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
        #track_vel = self.track_vel + 0.25*np.sin(self.step_no)
        track_vel = self.track_vel
        err_vel = np.abs(track_vel - realized_vel)
        err_vel = np.clip(err_vel,0,0.5)
        norm_err_vel = (err_vel - 0)/(0.5) ##   << -------------- Normalized Linear Vel

        '''
        # Lane centering reward params
        img, resX, resY = self.sim.getVisionSensorCharImage(self.visionSensorHandle)
        self.img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        self.lane_center()
        err_track = np.abs(self.error)
        if err_track > 125:
            err_track = 125   
        else:
            pass
        norm_err_track = (err_track - 0)/125 ##   << -------------- Normalized Lane centering
        
        # Preview Maximizing error

        pix_sum = np.sum(self.obs)
        pix_sum = pix_sum/(255*96*320)
        '''

        # Angular velocity reward params
        Gyro_Z = self.sim.getFloatSignal("myGyroData_angZ")
        if Gyro_Z:
            err_effort = np.abs(Gyro_Z) 
        else:
            err_effort = np.abs(0)     
        norm_err_eff = (err_effort)/0.6 ##   << -------------- Normalized angular velocity

        # Path Tracking Reward Parameters
        self.arc_length() #returns pose tracking aswell
        err_pth = np.clip(self.path_track_err,0,1)
        norm_err_path = (err_pth)/1

        # Minimize curvature reward paramas
        kappa = abs((err_effort)/realized_vel)
        kappa = np.clip(kappa,0,6)
        norm_kappa = kappa/6

        # Pose Tracking Reward Parameters
        #err_pose = np.clip(self.pose_track_err,0,3.14)
        #norm_err_pose = (err_pose)/3.14
        
        #Camera effort
        #cam_jt_eff = self.camActErr
        #cam_jt_eff = np.clip(cam_jt_eff,0,0.7535)
        #norm_cam_jt_eff  = (cam_jt_eff)/0.7535

        # Total reward
        self.rew = (1 - norm_err_vel)**2 +(1- norm_err_eff)**2 + (1- norm_err_path)**2 + (1 - self.pose_track_err)**2 + (1 - norm_kappa)**2
        self.rew = np.float64(self.rew)


    def Logger(self):
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



'''
Environment Validation code

#Comment out three lines to validated environment

from stable_baselines3.common.env_checker import check_env
env = HuskyCPEnvCone()
check_env(env)
'''

   
'''
        
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
'''
