import numpy as np
from matplotlib import pyplot as plt

sim_posX_RT = np.genfromtxt("x_pos_rt.csv")
sim_posY_RT = np.genfromtxt("y_pos_rt.csv")
sim_posX_ = np.genfromtxt("x_pos_.csv")
sim_posY_ = np.genfromtxt("y_pos_.csv")

#sim_vel_xRT = np.genfromtxt("x_vel_rt.csv")
#sim_ang_zRT = np.genfromtxt("z_ang_rt.csv")

#Regural
sim_vel_x = np.genfromtxt("x_vel.csv")
sim_vel_x_DS = sim_vel_x[::2]
sim_ang_z = np.genfromtxt("z_ang.csv")
sim_ang_z_DS = sim_ang_z[::2]

########Extras#########
sim_vel_x2 = np.genfromtxt("x_vel2.csv")
sim_vel_x2_DS = sim_vel_x2[::2]

sim_vel_x3 = np.genfromtxt("x_vel3.csv")
sim_vel_x3_DS = sim_vel_x3[::2]

sim_ang_z2 = np.genfromtxt("z_ang2.csv")
sim_ang_z2_DS = sim_ang_z2[::2]

########################3

'''
#Mujoco
sim_vel_xM = np.genfromtxt("x_vel_mujoco.csv")
sim_vel_xM_DS = sim_vel_xM[::2]
sim_ang_zM = np.genfromtxt("z_ang_mujoco.csv")
sim_ang_zM_DS = sim_ang_zM[::2]

#Bullet 2.83
sim_vel_xB = np.genfromtxt("x_vel_bullet283.csv")
sim_vel_xB_DS = sim_vel_xB[::2]
sim_ang_zB = np.genfromtxt("z_ang_bullet283.csv")
sim_ang_zB_DS = sim_ang_zB[::2]

#Bullet ODE
sim_vel_xO = np.genfromtxt("x_vel_ode.csv")
sim_vel_xO_DS = sim_vel_xO[::2]
sim_ang_zO = np.genfromtxt("z_ang_ode.csv")
sim_ang_zO_DS = sim_ang_zO[::2]
'''

#Command wheel velocity 
cmd_wheel_l_sim = np.genfromtxt("cmd_wheel_l.csv")
cmd_wheel_l_sim_DS = cmd_wheel_l_sim[::2]
cmd_wheel_r_sim = np.genfromtxt("cmd_wheel_r.csv")
cmd_wheel_r_sim_DS = cmd_wheel_r_sim[::2]


#Input wheel velocity 
wheel_l_sim = np.genfromtxt("wheel_l.csv")
wheel_l_sim_DS = wheel_l_sim[::2]
wheel_r_sim = np.genfromtxt("wheel_r.csv")
wheel_r_sim_DS = wheel_r_sim[::2]

#Realized wheel velocity 
rlz_wheel_l_sim = np.genfromtxt("rlz_wheel_l.csv")
rlz_wheel_l_sim_DS = rlz_wheel_l_sim[::2]
rlz_wheel_r_sim = np.genfromtxt("rlz_wheel_r.csv")
rlz_wheel_r_sim_DS = rlz_wheel_r_sim[::2]


real_vel = np.genfromtxt("data_sine_ip0.csv", delimiter=',')

fig0, (ax1, ax2) = plt.subplots(2, 1)
fig0.suptitle('Input velocities comparison')

#plt.figure("Linear Velocity")
ax1.plot(wheel_r_sim_DS)
#ax1.plot(cmd_wheel_r_sim_DS)
#ax1.plot(real_vel[1,:])
ax1.plot(real_vel[3,:])
ax1.legend(["R Wheel sim input","R Wheel real"])
ax1.set(ylabel='Rad per s')

#plt.figure("Angular Velocity")
ax2.plot(wheel_l_sim_DS)
#ax2.plot(cmd_wheel_l_sim_DS)
ax2.plot(real_vel[0,:])
#ax2.plot(real_vel[2,:])
ax2.legend(["L Wheel sim input","L Wheel real"])
ax2.set(ylabel='Rad per s')
#plt.show()

fig1, (ax3, ax4) = plt.subplots(2, 1)
fig1.suptitle('sim cmd-rlz velocities comparison')

#plt.figure("Linear Velocity")
ax3.plot(rlz_wheel_r_sim_DS)
ax3.plot(cmd_wheel_r_sim_DS)
ax3.legend(["R cmd","R rlz"])
ax3.set(ylabel='Rad per s')

#plt.figure("Angular Velocity")
ax4.plot(rlz_wheel_l_sim_DS)
ax4.plot(cmd_wheel_l_sim_DS)
ax4.legend(["L cmd","L rlz"])
ax4.set(ylabel='Rad per s')
#plt.show()


fig2, (ax5, ax6) = plt.subplots(2, 1)
fig2.suptitle('Model comparison')

#plt.figure("Linear Velocity")
ax5.plot(sim_vel_x_DS)
ax5.plot(sim_vel_x2_DS,'*')
#ax5.plot(sim_vel_x3_DS)
#ax5.plot(sim_vel_xB_DS)
#ax5.plot(sim_vel_xO_DS)
ax5.plot(real_vel[4,:])
#ax5.legend(["Body","GlobalM","Global","Real"])
ax5.set(ylabel='Linear Velocity')

#plt.figure("Angular Velocity")
ax6.plot(sim_ang_z_DS)
ax6.plot(sim_ang_z2_DS)
#ax6.plot(sim_ang_zB_DS)
#ax6.plot(sim_ang_zO_DS)
ax6.plot(-real_vel[5,:])
ax6.legend(["Mujoco","Real"])
ax6.set(ylabel='Angular Velocity')

plt.figure("Position")
plt.plot(sim_posX_,sim_posY_,'--')
plt.plot(sim_posX_RT,sim_posY_RT)



plt.show()
