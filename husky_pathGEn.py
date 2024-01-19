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
#from matplotlib.animation import FuncAnimation

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


print('Program started')



client = RemoteAPIClient('localhost',23004)
sim = client.getObject('sim')

ctrlPts = [0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,0,0,0,1]

scale = 5

wp = [0,0,0,0,0,0,1, 2*scale,-2*scale,0,0,0,0,1, 4*scale,-1*scale,0,0,0,0,1, 6*scale,-2*scale,0,0,0,0,1, 8*scale,0,0,0,0,0,1, 6*scale,2*scale,0,0,0,0,1, 4*scale,1*scale,0,0,0,0,1, 2*scale,2*scale,0,0,0,0,1]
#pathHandle = sim.createPath(wp, 2,100,1.0)

pathHandle = sim.getObject('/Path')

pathData = sim.unpackDoubleTable(sim.readCustomDataBlock(pathHandle, 'PATH'))

pathArray = np.array(pathData)
reshaped = np.reshape(pathArray,(100,7))
print(np.shape(reshaped))

defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# Run a simulation in stepping mode:
client.setStepping(True)
sim.startSimulation()


#Get object handles from CoppeliaSim handles
visionSensorHandle = sim.getObject('/Vision_sensor')

while (t:= sim.getSimulationTime()) < 60:

    img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
            # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
            # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
            # and color format is RGB triplets, whereas OpenCV uses BGR:
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0) #Convert to Grayscale

            # Current image
    cropped_image = img[288:480, 192:448] # Crop image to only to relevant path data (Done heuristically)
    im_bw = cv2.threshold(cropped_image, 125, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
    im_bw = np.frombuffer(im_bw, dtype=np.uint8).reshape(192, 256, 1) # Reshape to required observation size
    cv2.imshow("Image", img)
    cv2.imshow("Image Cropped", cropped_image)
    cv2.imshow("Image Input", im_bw)
    cv2.waitKey(1)

    sim.step()

sim.stopSimulation()

'''

fig1 = plt.figure()
plt.plot(reshaped[:,0], reshaped[:,1])
plt.show()

fig2 = plt.figure()
plt.plot(reshaped[:,3])
plt.plot(reshaped[:,4])
plt.plot(reshaped[:,5])
plt.plot(reshaped[:,6])
plt.show()
'''