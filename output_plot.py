import numpy as np
from matplotlib import pyplot as plt
import pickle


'''
Format:

varA = np.genfromtxt("pathA")
varB = np.genfromtxt("pathB")

plt.figure("Name")
plt.plot(varA)
plt.plot(varB)

plt.show()

'''
path = '/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/heat_map/all_final/'
specifier = 'three'

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


'''

V = np.genfromtxt("/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/heat_map/all_final/ten_actV.csv")
omega = np.genfromtxt("/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/heat_map/all_final/ten_actW.csv")
V2 = np.genfromtxt("/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/heat_map/all_final/ten_actV3.csv")
omega2 = np.genfromtxt("/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/heat_map/all_final/ten_actW2.csv")
'''


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
#f3.show()

plt.show()