#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import pathlib
from operator import itemgetter

import numpy as np
import pandas as pd
import pint

import matplotlib.pyplot as plt
import matplotlib.colors
import pint
import scipy.stats
import skimage.transform
import pyDOE


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


# In[2]:

bathy = 'gebco'
bathy_folder = f'/beegfs/deltares/{bathy}'
run_id = 'd'


# ## Grid 
# We pick a general grid of 256, 256 cells of 10x10 meter. These are constants.

# In[3]:


n = {"min": 256, "max": 256, "type": "constant"}
m = {"min": 256, "max": 256, "type": "constant"}
spacing = {"min": 10 * ureg.meter, "max": 10 * ureg.meter, 'type': "constant"}

grid = dict(n=n, m=m, dx=spacing, dy=spacing)
grid


# ## Waterlevel
# We generate water levels varying from 1 to 50 meters. This is enough for wave lengths of up to 100m (similar to ~700km fetch, 60km/h wind, for 30 hours) to not "feel the bottom". Bathymetries are subtracted from this level. 
# 
# 

# In[4]:


water_level = {"$\eta$": {"min": 1 * ureg.meter, "max": 50 * ureg.meter, "type": "uniform"}}


# ## Waves 
# We take a jonswap spectrum where we only vary the modal frequency ($\omega$) and the wave angle ($\theta_{wave}$). We compute $\omega$ based the significant wave height ($\zeta$). 

# In[5]:


# TODO: Hm0 (modus significant wave height)
# TODO: Wave Period 
# TODO: check wave steilheid als invoer
# TODO:  check distribution from measurements


# def wave_height_to_omega(wave_height):
#     g = Q_(9.81, 'm/s^2')
#     omega = 0.4 * np.sqrt(g / wave_height)
#     return omega
    
# waves['$\omega$'] = {
#     "min": wave_height_to_omega(waves['$\zeta$']['min']), 
#     "max": wave_height_to_omega(waves['$\zeta$']['max']),
#     "type": "uniform"
# }
waves = {
    "$\zeta$": {"min": Q_(1, ureg.meter),  "max": Q_(20, ureg.meter), "type": "uniform"},
    '$\theta_{wave}$': {"min": Q_(0, ureg.radian), "max": Q_(2 * np.pi, ureg.radian), "type": "circular"}
}

waves


# ## Bathymetry
# For the bathymetry we have a collection of bathymetries with varying slope and another collection of random sampled bathymetries of around the world. 

# In[6]:


bathy_list = list(pathlib.Path(bathy_folder).glob('**/*.tif'))

bathymetry = {
    "$bathy_i$": {"min": 0, "max": len(bathy_list)-1, "type": "resample"}
}

# "$\theta_{bathy, x,y}$": {"min": Q_(0, 'rad'), "max": Q_(2 * np.pi, 'rad'), "type": "skip"}


# In[7]:


parameters = pd.DataFrame(dict(**grid, **waves, **water_level, **bathymetry)).T


# In[8]:


parameters


# In[9]:


def row_to_N(row):
    min = None
    if isinstance(row['min'], pint.Quantity):
        min = row['min'].m
    else:
        min = row['min']
    if isinstance(row['max'], pint.Quantity):
        max = row['max'].m
    else:
        max = row['max']
    return matplotlib.colors.Normalize(min, max, clip=True)

parameters['N'] = parameters.apply(row_to_N, axis=1)


# In[10]:



stochastic = parameters.query('type in ("uniform", "circular", "resample")')
stochastic


n = stochastic.shape[0]
samples = 3000
# take samples from the cube
cube_lhs = pyDOE.lhs(n, samples=samples)


# In[11]:


# all boundaries in normalized space
options = [[0, 1] for i in range(n)]
bounds = list(itertools.product(*options))

# do bounds first and then the filled cube
filled_cube = np.r_[bounds, cube_lhs]


# In[13]:




samples = {}
for i, (variable, N) in enumerate(stochastic.N.items()):
    normalized_sample = filled_cube[:, i]
    sample = N.inverse(normalized_sample)
    samples[variable] = sample
examples = pd.DataFrame(samples)
examples['$bathy_i$'] = np.int32(examples['$bathy_i$'])
examples['bathy_file'] = itemgetter(*examples['$bathy_i$'].tolist())(bathy_list)

    
    


# In[14]:


import rasterio

bathy_arrays = []
for i, file in enumerate(examples['bathy_file']):    
    src = rasterio.open(examples['bathy_file'][0])
    array = src.read(1)
    array -= array.mean()
    bathy_arrays.append(array)
    
examples['bathy'] = bathy_arrays


# In[20]:


#Write hdf5

examples.to_hdf(f'/beegfs/deltares/runs/bathy-{bathy}-{run_id}-runs.h5', 'runs')


# In[21]:


examples

