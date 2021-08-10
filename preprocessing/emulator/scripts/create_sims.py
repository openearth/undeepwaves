#!/usr/bin/env python

import os
import itertools
from shutil import copyfile, copymode
from decimal import Decimal, ROUND_HALF_UP
import pathlib

# external
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# our modules
## load functions
from py_functions import render_template, create_bathy

# =============================================================================
# 1. input


# =============================================================================

## template path
path_template        = os.path.join('template')
bathy = 'gebco'
run_id = 'd'
## sims path
path_sims            = f'/beegfs/deltares/runs/bathy-{bathy}-{run_id}-runs'
# runs file
runs_file = f'/beegfs/deltares/runs/bathy-{bathy}-{run_id}-runs.h5'

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
    if '$\theta_{bathy, z}$' in item:
        bathy_angle = float(item['$\theta_{bathy, z}$'])
    else:
        # bathy number
        bathy_angle = ii


    ## sim name
    if water_level>0:
        fname    = 'U{:02.0f}D{:03.0f}Lp{:03.0f}Tz{}{}'.format(wave_height, Decimal(wave_dir).to_integral_value(ROUND_HALF_UP), water_level*100, bathy_angle, variant)
    else:
        fname    = 'U{:02.0f}D{:03.0f}Lm{:03.0f}Tz{}{}'.format(wave_height, Decimal(wave_dir).to_integral_value(ROUND_HALF_UP), -1*water_level*100, bathy_angle, variant)
    

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

    Z = item['bathy']
    np.savetxt(os.path.join(path_sim, 'bathy.dep'), Z)
    ## add sim to batch script
    string = string + 'cd {}\nsbatch swan4131A1c7.sh {} \ncd ..\n'.format(fname,fname)
    ## create batch script with N_batch

## write file
render_template({'output_path':path_sims,
                 'fname':'run_all',
                 'extension':'sh',
                 'header':'--',
                 'string':string},template_run)


    
    
