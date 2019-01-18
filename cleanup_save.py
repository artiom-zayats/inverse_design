import numpy as np
import pandas as pd
import os
import pdb

print("Starting FileLoading")

#Load data

df = pd.read_csv(os.path.join("data","big.csv"))



#parsing info, deleting extra stuff
#df['sim_mode'] = df.CSCS.str.split("-").str.get(0)
df['sim_wavelength'] = df.CSCS.str.split("-").str.get(1)
df['latice_basis'] = df.CSCS.str.split("-").str.get(2).str.split("/").str.get(0)
#pdb.set_trace()
#df['material']= df.CSCS.str.split("-").str.get(2).str.split("/").str.get(1).str.split('=').str.get(0)
df['material_thickness'] = df.CSCS.str.split("-").str.get(2).str.split("/").str.get(1).str.split('=').str.get(1).str.split(':').str.get(0)
df['patter_shape'] = test = df.CSCS.str.split("-").str.get(2).str.split("/").str.get(1).str.split('=').str.get(1).str.split(':').str.get(1).str.get(0)
Temp = df.CSCS.str.split("-").str.get(2).str.split("/").str.get(1).str.split('=').str.get(1).str.split(':').str.get(1).str[1::]
Temp = Temp.str.replace("(","").str.replace(")","")
#df['pattern_positionX'] = Temp.str.split(',').str[0]
#df['pattern_positionY'] = Temp.str.split(',').str[1]
df['radius'] = Temp.str.split(',').str[2]

df = df[['transmission','phase','sim_wavelength','material_thickness','radius','latice_basis']].copy()
#renaming columns
df.columns = ['A','phi','lambda','T','R','P']
#pdb.set_trace()

#splitting data into wavelength
wavelengths = set(df['lambda'])
sorted_wavelength = sorted(wavelengths)
data1 = df[df['lambda']==sorted_wavelength[0]].reset_index(drop = True) #0.46
data2 = df[df['lambda']==sorted_wavelength[1]].reset_index(drop = True) #0.54
data3 = df[df['lambda']==sorted_wavelength[2]].reset_index(drop = True) #0.7


print("Merging")
#merge and rename1
combined = pd.merge(data1, data2,  how='inner', on=['T','R','P'])
combined.rename(index=str,columns = {'A_x':'A_1','A_y':'A_2','phi_x':'phi_1','phi_y':'phi_2','lambda_x':'lambda_1','lambda_y':'lambda_2'},inplace = True)
#merge and rename2
combined = pd.merge(combined, data3,  how='inner', on=['T','R','P'])
combined.rename(index=str,columns = {'A':'A_3','phi':'phi_3','lambda':'lambda_3'},inplace = True)

combined['P'] = combined['P'].str.replace("(","").str.replace(")","")

combined['P']=combined['P'].str.split(',').str[0]
#combined['P_2']=combined['P'].str.split(',').str[3]
#combined.drop(columns='P',inplace = True)
combined.drop_duplicates(inplace=True)
combined.reset_index(drop=True,inplace = True)



#splitting test
np.random.seed(1003)
#remove 5% for TEST
msk = np.random.rand(len(combined)) < 0.95

df_train = combined[msk]

df_test = combined[~msk]

df_test.to_pickle(os.path.join("data","test.pkl"))

print("File test.pkl saved")

#remove 10% of train to DEV
msk2 = np.random.rand(len(df_train)) < 0.90

df_dev = df_train[~msk2]
df_train = df_train[msk2]


#Save to pickfile
df_train.to_pickle(os.path.join("data","train.pkl"))
df_dev.to_pickle(os.path.join("data","dev.pkl"))




print("File train.pkl saved")
print("File dev.pkl saved")