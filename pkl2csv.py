
read_path = '/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf_all/'
#save_path = '/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf/csv'

import pickle as pickle
import pandas as pd
#with open("file.pkl", "rb") as f:
#    object = pkl.load(f)


for x in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    pass

    specifier = 'vp_'+ str(int(x*100))
    with open(read_path + specifier + "_err_vel_norm", "rb") as fp:   # Unpickling
        obj = pickle.load(fp)

    df = pd.DataFrame(obj)
    df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf_all/csv/'+ specifier + '_err_vel_norm.csv')

    with open(read_path + specifier+ "_err_feat_norm", "rb") as fp:   # Unpickling
        obj = pickle.load(fp)
    df = pd.DataFrame(obj)
    df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf_all/csv/'+ specifier + '_err_feat_norm.csv')

    with open(read_path + specifier+ "_err_omega_norm", "rb") as fp:   # Unpickling
        obj = pickle.load(fp)
    df = pd.DataFrame(obj)
    df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf_all/csv/'+ specifier + '_err_omega_norm.csv')





'''

specifier = 'vp_95'


with open(read_path + specifier + "_actV", "rb") as fp:   # Unpickling
    obj = pickle.load(fp)
df = pd.DataFrame(obj)
df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf/csv/'+ specifier + '_actV.csv')


with open(read_path + specifier+ "_actW", "rb") as fp:   # Unpickling
    obj = pickle.load(fp)
df = pd.DataFrame(obj)
df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf/csv/'+ specifier + '_actW.csv')

with open(read_path + specifier + "_err_vel", "rb") as fp:   # Unpickling
    obj = pickle.load(fp)
df = pd.DataFrame(obj)
df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf/csv/'+ specifier + '_err_vel.csv')

with open(read_path + specifier + "_err_vel_norm", "rb") as fp:   # Unpickling
    obj = pickle.load(fp)
df = pd.DataFrame(obj)
df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf/csv/'+ specifier + '_err_vel_norm.csv')

with open(read_path + specifier+ "_err_feat", "rb") as fp:   # Unpickling
    obj = pickle.load(fp)
df = pd.DataFrame(obj)
df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf/csv/'+ specifier + '_err_feat.csv')

with open(read_path + specifier+ "_err_feat_norm", "rb") as fp:   # Unpickling
    obj = pickle.load(fp)
df = pd.DataFrame(obj)
df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf/csv/'+ specifier + '_err_feat_norm.csv')

with open(read_path + specifier+ "_err_omega", "rb") as fp:   # Unpickling
    obj = pickle.load(fp)
df = pd.DataFrame(obj)
df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf/csv/'+ specifier + '_err_omega.csv')

with open(read_path + specifier+ "_err_omega_norm", "rb") as fp:   # Unpickling
    obj = pickle.load(fp)
df = pd.DataFrame(obj)
df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf/csv/'+ specifier + '_err_omega_norm.csv')

with open(read_path + specifier+ "_rel_vel_lin", "rb") as fp:   # Unpickling
    obj = pickle.load(fp)
df = pd.DataFrame(obj)
df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf/csv/'+ specifier + '_rel_vel_lin.csv')

with open(read_path + specifier+ "_rel_vel_ang", "rb") as fp:   # Unpickling
    obj = pickle.load(fp)
df = pd.DataFrame(obj)
df.to_csv(r'/home/asalvi/code_workspace/Husky_CS_SB3/csv_data/vel_perf/csv/'+ specifier + '_rel_vel_ang.csv')

'''