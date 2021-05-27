# This script is used for processing historical/baseline WM-run (without ABM) to determine historical water supply availability,
# used for bias-correction of water availability at grid cells

import os
import pandas as pd
import numpy as np
import xarray as xr
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
pd.set_option('display.expand_frame_repr', False)

# Load NLDAS grid cells table for study area
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_netcdf')
nldas = pd.read_csv('nldas.txt')
nldas_lookup = pd.read_csv('nldas_states_counties_regions.csv')

# Load ABM nldas IDs
with open('nldas_ids.p', 'rb') as fp:
    nldas_ids = pickle.load(fp)

# start year
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

# start_year = 2010
# end_year = 2010
start_year = 2000
end_year = 2000

#os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_netcdf\\revised_demand_hist_results')
os.chdir('C:\\Users\\yoon644\\Desktop\\ABM runs 2050 v3')

# loop through .nc files and extract data
first = True
for year in range(start_year, end_year+1):
    for month in months:
        #dataset_name = 'new_demand_test.mosart.h0.' + str(year) + '-' + month + '.nc'
        dataset_name = 'warmup_1990_2000.mosart.h0.' + str(year) + '-' + month + '.nc'
        ds = xr.open_dataset(dataset_name)
        df = ds.to_dataframe()
        df_merge = pd.merge(df, nldas, how='left', left_on=['lat', 'lon'], right_on=['CENTERY', 'CENTERX'])
        df_select = df_merge[['NLDAS_ID', 'WRM_DEMAND0','WRM_SUPPLY','WRM_DEFICIT','WRM_STORAGE','GINDEX','RIVER_DISCHARGE_OVER_LAND_LIQ']]
        df_select['year'] = year
        df_select['month'] = int(month)
        if first:
            df_all = df_select
            first = False
        else:
            df_all = pd.concat([df_all, df_select])

# calculate average across timesteps
df_pivot = pd.pivot_table(df_all, index=['NLDAS_ID','GINDEX'], values=['WRM_SUPPLY', 'WRM_STORAGE','RIVER_DISCHARGE_OVER_LAND_LIQ'], aggfunc=np.mean) # units will be average monthly (m3/s)
df_pivot = df_pivot.reset_index()

# convert units from m3/s to acre-ft/yr
df_pivot['WRM_SUPPLY_acreft'] = df_pivot['WRM_SUPPLY'] * 60 * 60 * 24 * 30.42 * 12 / 1233.48

# join .nc calculations to ABM NLDAS table
abm_supply_avail = df_pivot[df_pivot['NLDAS_ID'].isin(nldas_ids)].reset_index()
abm_demand = pd.read_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_netcdf\\hist_demand_for_ncdf_nirnon0v2.csv')
abm_supply_avail = pd.merge(abm_supply_avail, abm_demand, how='left',on='NLDAS_ID')
abm_supply_avail['sw_irrigation_vol'] = abm_supply_avail['sw_irrigation_m3s'] * 25583.64
abm_supply_avail['sw_avail_bias_corr'] = abm_supply_avail['sw_irrigation_vol'] - abm_supply_avail['WRM_SUPPLY_acreft']
abm_supply_avail = abm_supply_avail[['NLDAS_ID', 'sw_irrigation_vol', 'WRM_SUPPLY_acreft', 'sw_avail_bias_corr','RIVER_DISCHARGE_OVER_LAND_LIQ']]
abm_supply_avail.rename(columns={'WRM_SUPPLY_acreft': 'WRM_SUPPLY_acreft_OG'}, inplace=True)
abm_supply_avail.rename(columns={'RIVER_DISCHARGE_OVER_LAND_LIQ': 'RIVER_DISCHARGE_OVER_LAND_LIQ_OG'}, inplace=True)
abm_supply_avail.to_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_netcdf\\hist_avail_bias_correction_20201102.csv')

# water_constraints_by_farm_test = abm_supply_avail['WRM_SUPPLY_acreft'].to_dict()

# calculate average upstream storage values
ds = xr.open_dataset('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\IM3\\WM Flag Tests\\US_reservoir_8th_NLDAS3_updated_CERF_Livneh_naturalflow.nc')
dams = ds["DamInd_2d"].to_dataframe()
dams = dams.reset_index()

dep = ds["gridID_from_Dam"].to_dataframe()
dep = dep.reset_index()
dep_id = ds["unit_ID"].to_dataframe()

dep_merge = pd.merge(dep, dams, how='left', left_on=['Dams'], right_on=['DamInd_2d'])

df_pivot = pd.merge(df_pivot, nldas, how='left', on='NLDAS_ID')

dep_merge = pd.merge(dep_merge, df_pivot[['NLDAS_ID','CENTERX','CENTERY','WRM_STORAGE']], how='left', left_on=['lat','lon'], right_on=['CENTERY','CENTERX'])

aggregation_functions = {'WRM_STORAGE': 'sum'}
dep_merge = dep_merge.groupby(['gridID_from_Dam'], as_index=False, dropna=False).aggregate(aggregation_functions)
dep_merge.rename(columns={'WRM_STORAGE': 'STORAGE_SUM'}, inplace=True)

df_pivot = pd.merge(df_pivot, dep_merge, how='left', left_on=['GINDEX'], right_on=['gridID_from_Dam'])
storage_sum = df_pivot[df_pivot['NLDAS_ID'].isin(nldas_ids)].reset_index()
storage_sum = storage_sum[['NLDAS_ID','STORAGE_SUM']]
storage_sum = storage_sum.fillna(0)
storage_sum.rename(columns={'STORAGE_SUM': 'STORAGE_SUM_OG'}, inplace=True)

storage_sum.to_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_netcdf\\hist_dependent_storage.csv')