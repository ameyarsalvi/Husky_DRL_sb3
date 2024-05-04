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



client = RemoteAPIClient('localhost',23006)
sim = client.getObject('sim')


#ctrlPts = [0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,0,1]

#scale = 5

#wp = [0,0,0,0,0,0,1, 2*scale,-2*scale,0,0,0,0,1, 4*scale,-1*scale,0,0,0,0,1, 6*scale,-2*scale,0,0,0,0,1, 8*scale,0,0,0,0,0,1, 6*scale,2*scale,0,0,0,0,1, 4*scale,1*scale,0,0,0,0,1, 2*scale,2*scale,0,0,0,0,1]
#pathHandle = sim.createPath(wp, 2,100,1.0)


pathHandle = sim.getObject('/PathR')

pathData = sim.unpackDoubleTable(sim.readCustomDataBlock(pathHandle, 'PATH'))

pathArray = np.array(pathData)
print(np.shape(pathData))


reshaped = np.reshape(pathArray,(250,7))
print(np.shape(reshaped))

print(reshaped[:,0])
print(reshaped[:,1])
print(reshaped[:,2])
print(reshaped[:,3])
print(reshaped[:,4])
print(reshaped[:,5])
print(reshaped[:,6])


#rev_pathL = [x_rev,y_rev]
#print(np.shape(rev_pathL))


#import pandas as pd 
#df = pd.DataFrame(reshaped)
#df.to_csv("/home/asalvi/code_workspace/Husky_CS_SB3/HuskyModels/path2/pathR.csv", header=False, index=False)

#fig1 = plt.figure()
#plt.plot(reshaped[:,0], reshaped[:,1])
#plt.plot(rev_pathL[0],rev_pathL[1],'*')
#plt.show()


#fig2 = plt.figure()
#plt.plot(reshaped[:,3])
#plt.plot(reshaped[:,4])
#plt.plot(reshaped[:,5])
#plt.plot(reshaped[:,6])
#plt.show()


############### Cone placement



cone = []
#primary_cone = sim.getObject('/Cone[0]')
#print(primary_cone)


#dummy = [None] * 100

for x in range(250):
        #print([cone_0])
        #cone_0 = sim.getObject(cone_0)
        #str_pt = '/Cone[' + str(int(x)) + ']' 
    #str_pt = '/Cone[0]' 
        #print(str_pt)
    cone = sim.getObject('/Cone2[0]')
    cone = sim.copyPasteObjects([cone],1)
    next_cone = sim.getObject('/Cone2[' + str(int(x+1)) + ']')
    #pose = sim.getObjectPose(cone0 , sim.handle_world)
        #pose[0] = reshaped[x,0]
        #pose[1] = reshaped[y,1]
        #cone_hand = sim.getObject('/Cone')
    sim.setObjectPose(next_cone, [reshaped[x,0],reshaped[x,1],0.0254,0,0,0,1], sim.handle_world)
    del cone
        #del pose




'''
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Run a simulation in stepping mode:
client.setStepping(True)
sim.startSimulation()


#Get object handles from CoppeliaSim handles
visionSensorHandle = sim.getObject('/Vision_sensor')

while (t:= sim.getSimulationTime()) < 600:[reshaped[i,0], reshaped[i+1,0]

    img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
            # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
            # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
            # and color format is RGB triplets, whereas OpenCV uses BGR:
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale

            # Current image
    #cropped_image = img[288:480, 192:448] # Crop image to only to relevant path data (Done heuristically)
    cropped_image = img[270:480, 0:640] # Crop image to only to relevant path data (Done heuristically)
    im_bw = cv2.threshold(cropped_image, 125, 200, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    im_bw = cv2.threshold(im_bw, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    
    im_bw_error = cv2.threshold(cropped_image, 225, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    im_bw_input = np.frombuffer(im_bw, dtype=np.uint8).reshape(210, 640, 1) # Reshape to required observation size
    cv2.imshow("Image", img)
    cv2.imshow("Image Cropped", cropped_image)
    cv2.imshow("Image input", im_bw)
    cv2.imshow("Image error", im_bw_error)
    #cv2.imshow("Image Input", im_bw)
    

    ######### Test control (P) ###########
    # calculate moments of binary image
    

    def cent_find(image_ip):
        M = cv2.moments(image_ip)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        return 320-cX
    #M2 = cv2.moments(im_bw_right)
 
    # calculate x,y coordinate of center
    

    # calculate x,y coordinate of center
    #if M2["m00"] != 0:
    #    cX2 = int(M2["m10"] / M2["m00"])
    #    cY2 = int(M2["m01"] / M2["m00"])
    #else:
    #    cX2, cY2 = 0, 0

 
    # put text and highlight the center
    #cv2.circle(im_bw, (cX1, cY1), 5, (0, 0, 0), -1)
    #cv2.putText(im_bw, "centroid left", (cX1 - 25, cY1 - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    #cv2.circle(im_bw, (cX2, cY2), 5, (0, 0, 0), -1)
    #cv2.putText(im_bw, "centroid right", (cX2 - 25, cY2 - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

 
    # display the image
    #cv2.imshow("Image0", im_bw0)
    #cv2.imshow("Image1", im_bw1)
    #cv2.imshow("Image2", im_bw2)
    #cv2.imshow("Image3", im_bw3)
    #cv2.imshow("Image4", im_bw4)
    #cv2.imshow("Image with centroid Right", im_bw_right)

    cv2.waitKey(1)

    im_bw_error = im_bw_error
    error = cent_find(im_bw_error)
    #error2 = np.abs(160-cX2)
    #error = (error1 +error2)/2
    p_gain = 0.005
    V = 0.3
    omega = p_gain*error
    A = np.array([[0.081675,0.081675],[-0.1081,0.1081]]) 
    velocity = np.array([[V],[omega]])
    phi_dots = np.matmul(inv(A),velocity)
    phi_dots = phi_dots.astype(float)
    Left = phi_dots[0].item()
    Right = phi_dots[1].item()

    fl_w = sim.getObject('/flw')
    fr_w = sim.getObject('/frw')
    rr_w = sim.getObject('/rrw')
    rl_w = sim.getObject('/rlw')

    sim.setJointTargetVelocity(fl_w, Left)
    sim.setJointTargetVelocity(fr_w, Right)
    sim.setJointTargetVelocity(rl_w, Left)
    sim.setJointTargetVelocity(rr_w, Right)

    sim.step()

sim.stopSimulation()
'''



