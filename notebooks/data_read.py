import numpy as np
import pandas as pd
import xarray as xr

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

source_path = Path('P:/11207539-001-undeepwaves/')


#Initialize dictionaries and store paths
s = 0
paths = {}
errors = []
res = {}
names = {}
param = {}
for a in source_path.glob("*"):
    if a.is_dir() and str(a.parts[-1][0:5]) == 'bathy':
        paths[str(a.parts[-1])] = a
        res[str(a.parts[-1])] = {}
        names[str(a.parts[-1])] = {}
        s += 1
      
        
#Add first 200 datasets and filenames of each directory to dictionary  
for p in paths:
    s = 0
    for a in paths[p].glob("*/*"):
        if a.is_dir():
            string = a.parts[-2]
            if s >= 0 and s <= 200:
                try:
                    res[p][s] = xr.open_dataset(a.joinpath(string+'.nc'),engine='scipy')
                    names[p][s] = string
                    s += 1
                except:
                    errors.append(str(a))
                    s += 1
            #elif s<201:
                #s += 1
            else:
                break
            
#Store parameters dataframes
params = {}
for k in res:
    params[k] = pd.read_hdf(source_path.joinpath(k+'.h5'))

#P:\11207539-001-undeepwaves\bathy-gebco-b-runs\7278e0a2-d885-4726-a432-12cb491dd30e\results error

#%% saving
np.save('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/Results1.npy',res)
np.save('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/Names1.npy',names)
np.save('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/Parameters.npy',params)


#%% Loading
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
source_path = Path('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results')
source_path2 = Path('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results')

#source_path2 = Path('P:/11207539-001-undeepwaves/')
results1 = np.load(source_path.joinpath('Results1.npy'),allow_pickle=True).item()
results2 = np.load(source_path.joinpath('Results2.npy'),allow_pickle=True).item()
results3 = np.load(source_path.joinpath('Results3.npy'),allow_pickle=True).item()
results4 = np.load(source_path.joinpath('Results4.npy'),allow_pickle=True).item()
results5 = np.load(source_path.joinpath('Results5.npy'),allow_pickle=True).item()

names1 = np.load(source_path.joinpath('Names1.npy'),allow_pickle=True).item()
names2 = np.load(source_path.joinpath('Names2.npy'),allow_pickle=True).item()
names3 = np.load(source_path.joinpath('Names3.npy'),allow_pickle=True).item()
names4 = np.load(source_path.joinpath('Names4.npy'),allow_pickle=True).item()
names5 = np.load(source_path.joinpath('Names5.npy'),allow_pickle=True).item()

params = np.load(source_path.joinpath('Parameters.npy'),allow_pickle=True).item()

DFEmoda = pd.read_pickle(source_path2.joinpath('DataEmoda.npy'))
DFEmodb = pd.read_pickle(source_path2.joinpath('DataEmodb.npy'))
DFEmodc = pd.read_pickle(source_path2.joinpath('DataEmodc.npy'))
DFEmodd = pd.read_pickle(source_path2.joinpath('DataEmodd.npy'))
DFEmode = pd.read_pickle(source_path2.joinpath('DataEmode.npy'))

DFGeba = pd.read_pickle(source_path2.joinpath('DataGeba.npy'))
DFGebb = pd.read_pickle(source_path2.joinpath('DataGebb.npy'))
DFGebc = pd.read_pickle(source_path2.joinpath('DataGebc.npy'))
DFGebd = pd.read_pickle(source_path2.joinpath('DataGebd.npy'))
DFGebe = pd.read_pickle(source_path2.joinpath('DataGebe.npy'))
DFGebf = pd.read_pickle(source_path2.joinpath('DataGebf.npy'))

DFSchema = pd.read_pickle(source_path2.joinpath('DataSchema.npy')) #Augment schematic bathymetry

#%% Reordering dataframes to have match between parameters and results


names = [names1,names2,names3,names4,names5]
names_full = {}
for key in names1:
    temp = []    
    for i in names:
        for j in i[key]:
            temp.append(i[key][j])
    names_full[key] = temp
    
# Reorder parameter dataframe to allign with results dataset
for key in names_full:
    dftemp = pd.DataFrame(names_full[key])
    dftemp = dftemp.sort_values(0)
    DfOrd = dftemp.index.values
    params[key] = params[key].sort_values('uuid')
    params[key] = params[key].set_index(DfOrd,drop=False)
    


#%% Combining Data

# Combine the dictionaries containing the results
merge = {}
for key in results1:
    merge[key] = {**results1[key], **results2[key], **results3[key], **results4[key], **results5[key]}


#%%  Get variables which we want to train from the datasets
source_path = Path("C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/")

# Parameter dataframe
Dummy = DFEmoda

# Get results from datasets and parameters from merge dictionary with 'folder' key
folder = 'bathy-schematic-a-runs'
Dummy = params[folder]


hs = {}
tm01 = {}
theta0 = {}
for i in merge[folder]:
    hs[i] = merge[folder][i].hs.data[0]
    tm01[i] = merge[folder][i].tm01.data[0]
    theta0[i] = merge[folder][i].theta0.data[0]
    # Print to check progress
    if i%100 == 0:
        print(i)

# Create dictionary with variables from output dataset
c = {}
d = {}
e = {}
for i in Dummy.index:
    c[i] = hs[i]
    d[i] = tm01[i]
    e[i] = theta0[i]

# Add dictionaries to parameter dataframe
Dummy = Dummy.assign(hs = c.values(), tm01 = d.values(), theta0 = e.values())

# Transpose of bathymetries to match orientation of the output variables
for i in Dummy.index:
    Dummy['bathy'][i] = Dummy['bathy'][i].T                

# Drop columns from dataframe
Dummy = Dummy.drop(['bathy_file','run_id','uuid','bathy_source'],axis=1)


Dummy.to_pickle(r'C:\Users\hemert\OneDrive - Stichting Deltares\Programmas\Data\Analysis Results\Dummy.npy')


#%% Convert angles to x and y direction components
Dummy = DFEmoda
theta_wavex = np.cos(Dummy['$\theta_{wave}$'])
theta_wavey = np.sin(Dummy['$\theta_{wave}$'])

theta0_rad = {}
theta0x = {}
theta0y = {}
for i in Dummy.index:
    theta0_rad[i] = np.deg2rad(Dummy['theta0'][i])
    theta0x[i] = np.cos(theta0_rad[i])
    theta0y[i] = np.sin(theta0_rad[i])

Dummy = Dummy.assign(theta_wavex = theta_wavex.values, theta_wavey = theta_wavey.values,
                     theta0x = theta0x.values(), theta0y = theta0y.values())
Dummy.to_pickle(r'C:\Users\hemert\OneDrive - Stichting Deltares\Programmas\Data\Analysis Results\DataEmoda.npy')








