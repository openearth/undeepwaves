#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import pathlib
from operator import itemgetter
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import pint
from tqdm.auto import tqdm

import click
import matplotlib.pyplot as plt
import matplotlib.colors
import pint
import rasterio
import scipy.stats
from shutil import copyfile, copymode
import skimage.transform
import pyDOE
import os

# our modules
## load functions
from py_functions import render_template, create_bathy


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity



@click.command()
@click.option('--bathy',
    required=True,
    type=click.Choice(['schematic', 'emodnet', 'gebco']),
    help="Choose bathy samples")
@click.option('--bathy_folder',help="Folder with bathy samples")
@click.option('-o', '--output', required=True, help="Output folder for swan runs")
@click.option('--id', required=True, help="Run id for set of runs")
@click.option('--samples', default=10, help="Number of runs", type=int)

# In[2]:




def main(bathy, bathy_folder, output, id, samples):
    # bathy schematic, gebco or emodnet
    run_id = id
    # bathy_folder = f'/beegfs/deltares/{bathy}'
    bathy_folder = f'{bathy_folder}/{bathy}'
    output_folder = f'{output}/bathy-{bathy}-{run_id}-runs.h5'



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


    if bathy in ['gebco', 'emodnet']:
        bathy_list = list(pathlib.Path(bathy_folder).glob('**/*.tif'))
        bathymetry = {
            "$bathy_i$": {"min": 0, "max": len(bathy_list)-1, "type": "resample"}
        }
    elif bathy == 'schematic':
        bathymetry = {
            "$\theta_{bathy, z}$": {"min": Q_(0, 'degree'), "max": Q_(10, 'degree'), "type": "uniform"}
        }
    else:
        print(f'unknown bathy type {bathy}')



    # In[7]:


    parameters = pd.DataFrame(dict(**grid, **waves, **water_level, **bathymetry)).T


    # In[8]:


    def create_bathy(rot_z):
        width = 256
        dx = 10
        op_by_adj = np.tan(np.deg2rad(rot_z)) 
        height = op_by_adj * np.arange(width) * dx
        z = np.tile(height, [width, 1])
        z = z - np.max(z)

        return z


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
    # samples = 10
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


    if bathy in ['gebco', 'emodnet']:
        examples['$bathy_i$'] = np.int32(examples['$bathy_i$'])
        examples['bathy_file'] = itemgetter(*examples['$bathy_i$'].tolist())(bathy_list)
        bathy_arrays = []
        for i, file in enumerate(examples['bathy_file']):
            print(f'bathymetry {i}, {file}')    
            src = rasterio.open(examples['bathy_file'][i])
            array = src.read(1)
            array -= np.nanmean(array)
            # fill nan with 999 this is execption value in swan
            array = np.nan_to_num(array, nan=999)
            print(array)
            bathy_arrays.append(array)

        examples['bathy'] = bathy_arrays

    if bathy == 'schematic':
        bathy_arrays = []
        for rot_z in examples["$\theta_{bathy, z}$"]:
            z = create_bathy(rot_z=rot_z)
            print(z.shape)
            bathy_arrays.append(z)
        examples['bathy'] = bathy_arrays


    # In[14]:

    # Add UUID

    import uuid
    examples['uuid'] = examples.apply(lambda _: uuid.uuid4(), axis=1)

    examples['run_id'] = run_id
    examples['bathy_source'] = bathy



    # In[20]:


    #Write hdf5

    examples.to_hdf(output_folder, 'runs')
    print(examples)

    # =============================================================================
    # 1. input


    # =============================================================================

    ## template path
    path_template        = os.path.join('template')
    ## sims path
    # path_sims            = f'/beegfs/deltares/runs/bathy-{bathy}-{run_id}-runs'
    path_sims = f'{output}/bathy-{bathy}-{run_id}-runs'
    # runs file
    # runs_file = f'/beegfs/deltares/runs/bathy-{bathy}-{run_id}-runs.h5'
    runs_file = output_folder

    ## variant
    variant = run_id


    ## if file exists overwrite
    overwrite            = False

    swan_template        = 'swan_template_V2.swn'

    # =============================================================================
    # 3. load combinations
    # =============================================================================
    df = pd.read_hdf(runs_file, 'runs')

    print(f'loaded {df.shape[0]} runs')

    # =============================================================================
    # 4. create sims
    # =============================================================================
    ## empty string for bath script
    string      = ''
    ## template file
    template        = os.path.join(path_template,swan_template)
    template_run    = os.path.join(path_template,'run_template.sh')


    ## make combinations
    for ii, item in tqdm(df.iterrows()):
        ## get conditions
        # convert to native python type (for compatibility with decimal)
        wave_dir    = float(item['$\theta_{wave}$'])
        wave_height  = float(item['$\zeta$'])
        water_level = float(item['$\eta$'])

        fname = str(item['uuid'])

        ## sim path
        path_sim = os.path.join(path_sims, fname)
        
        ## make dir
        if not os.path.exists(path_sim):
            os.makedirs(path_sim)
            os.makedirs(os.path.join(path_sim,'results'))
        else:
            print('path exist:',fname)

     
        # write parameters
        item.to_json(os.path.join(path_sim, 'parameters.json'), default_handler=str)
        
        ## data for template
        data = {'output_path':path_sim,
                'extension':'swn',
                'variant':variant,
                'fname':fname,
                'wind_speed': 0,
                'wind_dir': wave_dir,
                'wave_dir':wave_dir,
                'wave_height': wave_height,
                'wave_period':15,
                'water_level': water_level,
                'variant':variant
        }
        
        ## skip if file already exists
        if os.path.exists(os.path.join(path_sim, '{}.swn'.format(fname))) and overwrite:
            print('skipped {}'.format(fname))
            continue
        ## render template
        render_template(data,template)
        ## copy swan_run
        copyfile(os.path.join('template','swan4131A1c7.sh'), os.path.join(path_sim,'swan4131A1c7.sh'))
        copymode(os.path.join('template','swan4131A1c7.sh'), os.path.join(path_sim,'swan4131A1c7.sh'))
        copyfile(os.path.join('template','register_run.py'), os.path.join(path_sim,'register_run.py'))
        copyfile(os.path.join('template','points.txt'), os.path.join(path_sim,'points.txt'))

        
        Z = item['bathy']
        np.savetxt(os.path.join(path_sim, 'bathy.dep'), Z)
        ## add sim to batch script
        string = string + 'cd {}\nqsub swan4131A1c7.sh {} \ncd ..\n'.format(fname,fname)
        ## create batch script with N_batch

    ## write file
    render_template({'output_path':path_sims,
                     'fname':'run_all',
                     'extension':'sh',
                     'header':'--',
                     'string':string},template_run)


if __name__ == '__main__':
    main()

