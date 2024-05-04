import numpy as np
from matplotlib import pyplot as plt
import pickle


'''
Format (CSV):

varA = np.genfromtxt("pathA")
varB = np.genfromtxt("pathB")

plt.figure("Name")
plt.plot(varA)
plt.plot(varB)

plt.show()

Format (Pickle):

with open("pathA", "rb") as fp:   # Unpickling
    varA = pickle.load(fp)

plt.figure("Name")
plt.plot(varA)

plt.show()

'''


path = '/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/sharp/'
specifier = 'vp_75'


with open(path + specifier + "_actV", "rb") as fp:   # Unpickling
    V = pickle.load(fp)

with open(path + specifier+ "_actW", "rb") as fp:   # Unpickling
    omega = pickle.load(fp)


with open(path + specifier + "_err_vel_norm", "rb") as fp:   # Unpickling
    err_Vn = pickle.load(fp)

with open(path + specifier+ "_err_feat_norm", "rb") as fp:   # Unpickling
    err_Fn = pickle.load(fp)

with open(path + specifier+ "_rel_vel_lin", "rb") as fp:   # Unpickling
    rel_V = pickle.load(fp)

with open(path + specifier+ "_err_feat", "rb") as fp:   # Unpickling
    feat_tr = pickle.load(fp)



f1 = plt.figure("Actions")
plt.plot(V)
plt.plot(omega)
#f1.show()


f2 = plt.figure("Track Norm")
plt.plot(err_Fn)
plt.plot(err_Vn)
#f2.show()

f3 = plt.figure("Scatter")
plt.scatter(err_Fn,err_Vn)
plt.xlim(0, 1)
plt.ylim(0, 1)


print(np.shape(err_Fn))
print(np.shape(err_Vn))

f3 = plt.figure("Vel")
plt.plot(rel_V)

f4 = plt.figure("feat_tr")
plt.plot(feat_tr)


fig5, axs= plt.subplots(2, 2)
fig5.suptitle('Model Output : Vp75')

#plt state : Realized lin vel
axs[0,0].plot(rel_V)
axs[0,0].set(ylabel='Linear Velocity')

#plt state : Feat track err
axs[0,1].plot(feat_tr)
axs[0,1].set(ylabel='Lane Center')

#plt state : Actions
axs[1,0].plot(V)
axs[1,0].plot(omega)
axs[1,0].set(ylabel='Actions')

#plt state : Normalized Tracing err
axs[1,1].plot(err_Fn)
axs[1,1].plot(err_Vn)

plt.show()