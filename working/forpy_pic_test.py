from netCDF4 import Dataset
import os
import pandas as pd
import numpy as np

# Load NLDAS grid cells table for study area
nldas = pd.read_csv('nldas.txt')

# Pandas get MOSART-WM input data based on row and column
def get_data(row):
    try:
        return np.asscalar(data[row['NLDAS_Y']-1, row['NLDAS_X']-1])
    except IndexError:
        return 0

# Pandas get MOSART-WM output data based on row and column
def get_output(row):
    try:
        return np.asscalar(data[0, row['NLDAS_Y']-1, row['NLDAS_X']-1])
    except IndexError:
        return 0

# Load in dam locations to pandas dataframe
dataset_name = "US_reservoir_8th_NLDAS3_c20161220_updated_20170314.nc"
dataset = Dataset(dataset_name)
data = dataset.variables['DamID_Spatial']
nldas['DamID_Spatial'] = nldas.apply(get_data, axis=1)

# Load in dependency database to pandas dataframe
data = dataset.variables['num_Dam2Grid']
nldas['num_Dam2Grid'] = nldas.apply(get_data, axis=1)

# Load in storage capacity
data = dataset.variables['cap_mcm']
nldas['cap_mcm'] = nldas.apply(get_data, axis=1)

# Load in test modeled storage volume
dataset_name = "jimtest.mosart.h0.1987-12.nc"
dataset = Dataset(dataset_name)
data = dataset.variables['WRM_STORAGE']
nldas['WRM_STORAGE'] = nldas.apply(get_output, axis=1)
nldas['WRM_STORAGE'] = nldas['WRM_STORAGE'] / 1000000 # convert from m3 to MCM
nldas['perc_full'] = nldas['WRM_STORAGE'] / nldas['cap_mcm']