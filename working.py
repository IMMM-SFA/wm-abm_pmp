from pyomo.environ import *
from pyomo.opt import SolverFactory
import os
import pandas as pd
import numpy as np
import xarray as xr
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
import shutil
import netCDF4
import logging

year = '1981'
month = '01'

with open('data_inputs/nldas_ids.p', 'rb') as fp:
    nldas_ids = pickle.load(fp)

nldas = pd.read_csv('data_inputs/nldas.txt')

year_int = int(year)
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

with open('data_inputs/water_constraints_by_farm_v2.p', 'rb') as fp:
    #water_constraints_by_farm = pickle.load(fp)
    water_constraints_by_farm = pickle.load(fp, encoding='latin1')
water_constraints_by_farm = dict.fromkeys(water_constraints_by_farm, 9999999999)

## Read in Water Availability Files from MOSART-PMP

# pic_output_dir = '/pic/scratch/yoon644/csmruns/jimtest2/run/'
pic_input_dir = 'pic_input_test/'

# loop through .nc files and extract data
first = True
for m in months:
    # dataset_name = 'jim_abm_integration.mosart.h0.' + str(year-1) + '-' + m + '.nc'
    dataset_name = 'statemod.mosart.h0.' + str(year_int - 1) + '-' + m + '.nc'
    ds = xr.open_dataset(pic_input_dir + dataset_name)
    df = ds.to_dataframe()
    df_merge = pd.merge(df, nldas, how='left', left_on=['lat', 'lon'], right_on=['CENTERY', 'CENTERX'])
    df_select = df_merge[['NLDAS_ID', 'WRM_DEMAND0', 'WRM_SUPPLY', 'WRM_DEFICIT']]
    df_select['year'] = year_int
    df_select['month'] = int(m)
    if first:
        df_all = df_select
        first = False
    else:
        df_all = pd.concat([df_all, df_select])

# calculate average across timesteps
df_pivot = pd.pivot_table(df_all, index=['NLDAS_ID'], values=['WRM_SUPPLY'],
                          aggfunc=np.mean)  # units will be average monthly (m3/s)
df_pivot = df_pivot.reset_index()

# convert units from m3/s to acre-ft/yr
df_pivot['WRM_SUPPLY_acreft'] = df_pivot['WRM_SUPPLY'] * 60 * 60 * 24 * 30.42 * 12 / 1233.48
abm_supply_avail = df_pivot[df_pivot['NLDAS_ID'].isin(nldas_ids)].reset_index()
water_constraints_by_farm = abm_supply_avail['WRM_SUPPLY_acreft'].to_dict()

## Read in PMP calibration files
data_file = pd.ExcelFile("data_inputs/MOSART_WM_PMP_inputs_v1.xlsx")
data_profit = data_file.parse("Profit")
water_nirs = data_profit["nir_corrected"]
nirs = dict(water_nirs)

## C.1. Preparing model indices and constraints:
ids = range(592185)  # total number of crop and nldas ID combinations
farm_ids = range(53835)  # total number of farm agents / nldas IDs
with open('data_inputs/crop_ids_by_farm.p', 'rb') as fp:
    crop_ids_by_farm = pickle.load(fp)
with open('data_inputs/crop_ids_by_farm_and_constraint.p', 'rb') as fp:
    crop_ids_by_farm_and_constraint = pickle.load(fp)
with open('data_inputs/land_constraints_by_farm.p', 'rb') as fp:
    land_constraints_by_farm = pickle.load(fp, encoding='latin1')

# Load gammas and alphas
with open('data_inputs/gammas.p', 'rb') as fp:
    gammas = pickle.load(fp, encoding='latin1')
with open('data_inputs/net_prices.p', 'rb') as fp:
    net_prices = pickle.load(fp, encoding='latin1')

x_start_values = dict(enumerate([0.0] * 3))

## C.2. 2st stage: Quadratic model included in JWP model simulations
## C.2.a. Constructing model inputs:
##  (repetition to be safe - deepcopy does not work on PYOMO models)
fwm_s = ConcreteModel()
fwm_s.ids = Set(initialize=ids)
fwm_s.farm_ids = Set(initialize=farm_ids)
fwm_s.crop_ids_by_farm = Set(fwm_s.farm_ids, initialize=crop_ids_by_farm)
fwm_s.crop_ids_by_farm_and_constraint = Set(fwm_s.farm_ids, initialize=crop_ids_by_farm_and_constraint)
fwm_s.net_prices = Param(fwm_s.ids, initialize=net_prices, mutable=True)
fwm_s.gammas = Param(fwm_s.ids, initialize=gammas, mutable=True)
fwm_s.land_constraints = Param(fwm_s.farm_ids, initialize=land_constraints_by_farm, mutable=True)
fwm_s.water_constraints = Param(fwm_s.farm_ids, initialize=water_constraints_by_farm,
                                mutable=True)  # JY here need to read calculate new water constraints
fwm_s.xs = Var(fwm_s.ids, domain=NonNegativeReals, initialize=x_start_values)
fwm_s.nirs = Param(fwm_s.ids, initialize=nirs, mutable=True)


## C.2.b. 2nd stage model: Constructing functions:
def obj_fun(fwm_s):
    return 0.00001 * sum(sum(
        (fwm_s.net_prices[i] * fwm_s.xs[i] - 0.5 * fwm_s.gammas[i] * fwm_s.xs[i] * fwm_s.xs[i]) for i in
        fwm_s.crop_ids_by_farm[f]) for f in fwm_s.farm_ids)


fwm_s.obj_f = Objective(rule=obj_fun, sense=maximize)


def land_constraint(fwm_s, ff, ):
    return sum(fwm_s.xs[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.land_constraints[ff]


fwm_s.c1 = Constraint(fwm_s.farm_ids, rule=land_constraint)


def water_constraint(fwm_s, ff):
    return sum(fwm_s.xs[i] * fwm_s.nirs[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= \
           fwm_s.water_constraints[ff]


fwm_s.c2 = Constraint(fwm_s.farm_ids, rule=water_constraint)

## C.2.c Creating and running the solver:
opt = SolverFactory("ipopt", solver_io='nl')
results = opt.solve(fwm_s, keepfiles=False, tee=True)
print(results.solver.termination_condition)

## D.1. Storing main model outputs:
result_xs = dict(fwm_s.xs.get_values())


data_file=pd.ExcelFile("data_inputs/MOSART_WM_PMP_inputs_v1.xlsx")
data_profit = data_file.parse("Profit")
sd_no = len(farm_ids)
crop_types=[str(i) for i in list(pd.unique(data_profit["crop"]))]
crop_no=len(crop_types)
gamma1 = list(gammas.values())
with open('data_inputs/alphas.p', 'rb') as fp:
    alphas = pickle.load(fp, encoding='latin1')
alpha1 = alphas
obs_lu = dict(data_profit["area_irrigated"])


## D.2. Storing derived model outputs:
df_output=pd.DataFrame(index=range(0,(sd_no*crop_no)),columns=["farm_no","crop","land_use","irrigation","profit",
                                                                 "alpha","gamma","land_ratio_to_obs"])
df_output_mth=pd.DataFrame(index=range(1,1+(sd_no*12*crop_no)),columns=["farm_no","month","crop","land_use","irrigation","profit",
                                                                        "alpha","gamma"])
i=1
for sd in range(sd_no):
    for crp in range(crop_no):
        crop_id=crop_ids_by_farm[sd][crp]
        current_land_use=result_xs[crop_id] if result_xs[crop_id] else 0.0
        current_water_use=current_land_use*nirs[crop_id]
        current_profit=current_land_use*(net_prices[crop_id]+alphas[crop_id])
        current_alpha1=alpha1[crop_id]
        current_gamma1=gamma1[crop_id]
        land_ratio_to_obs = (current_land_use / obs_lu[crop_id]) if (obs_lu[crop_id] > 0.0) else int(obs_lu[crop_id] == round(current_land_use,0))
        df_output.loc[crop_id]=[farm_ids[sd],crop_types[crp],current_land_use,current_water_use,current_profit,
                                current_alpha1,current_gamma1,land_ratio_to_obs]
        # for mth in range(1,13):
        #     water_percent=float(data_seasons["water_percent"][(data_seasons["crop"]==crop_types[crp])&(data_seasons["month"]==mth)])
        #     current_water_use_mth=water_percent*current_water_use
        #     df_output_mth.loc[i]=[subdistricts[sd],mth,crop_types[crp],current_land_use,current_water_use_mth,current_profit,
        #                           current_alpha1,current_gamma1]
        #     i+=1

# JY store results into a pandas dataframe
results_pd = data_profit
results_pd = results_pd.assign(calc_area=result_xs.values())
results_pd = results_pd.assign(nir=nirs.values())
results_pd['calc_water_demand'] = results_pd['calc_area'] * results_pd['nir']
results_pivot = pd.pivot_table(results_pd, index=['nldas'], values=['calc_water_demand'],
                               aggfunc=np.sum)  # JY demand is order of magnitude low, double check calcs

# read a sample water demand input file
file = 'data_inputs/RCP8.5_GCAM_water_demand_1980_01_copy.nc'
with netCDF4.Dataset(file, 'r') as nc:
    # for key, var in nc.variables.items():
    #     print(key, var.dimensions, var.shape, var.units, var.long_name, var._FillValue)

    lat = nc['lat'][:]
    lon = nc['lon'][:]
    demand = nc['totalDemand'][:]

# read NLDAS grid reference file
df_grid = pd.read_csv('data_inputs/NLDAS_Grid_Reference.csv')

df_grid = df_grid[['CENTERX', 'CENTERY', 'NLDAS_X', 'NLDAS_Y', 'NLDAS_ID']]

df_grid = df_grid.rename(columns={"CENTERX": "longitude", "CENTERY": "latitude"})
df_grid['longitude'] = df_grid.longitude + 360

# match netCDF demand file and datagrame
mesh_lon, mesh_lat = np.meshgrid(lon, lat)
df_nc = pd.DataFrame({'lon': mesh_lon.reshape(-1, order='C'), 'lat': mesh_lat.reshape(-1, order='C')})
df_nc['NLDAS_ID'] = [
    'x' + str(int((row['lon'] - 235.0625) / 0.125 + 1)) + 'y' + str(int((row['lat'] - 25.0625) / 0.125 + 1)) for
    _, row in df_nc.iterrows()]
df_nc['totalDemand'] = 0

# use NLDAS_ID as index for both dataframes
df_nc = df_nc.set_index('NLDAS_ID', drop=False)
results_pivot = results_pivot.set_index('nldas', drop=False)

# read ABM values into df_nc basing on the same index
df_nc.loc[results_pivot.index, 'totalDemand'] = results_pivot.calc_water_demand.values

for month in months:
    str_year = str(year_int)
    new_fname = 'pic_output_test/test_' + str_year + '_' + month + '.nc'  # define ABM demand input directory
    #new_fname = 'pic_output_test/RCP8.5_GCAM_water_demand_' + str_year + '_' + month + '.nc'  # define ABM demand input directory
    shutil.copyfile(file, new_fname)
    demand_ABM = df_nc.totalDemand.values.reshape(len(lat), len(lon), order='C')
    with netCDF4.Dataset(new_fname, 'a') as nc:
        nc['totalDemand'][:] = np.ma.masked_array(demand_ABM, mask=nc['totalDemand'][:].mask)


max_land_constr = pd.read_csv('data_inputs/max_land_constr.csv')
temp_df = pd.DataFrame(nldas_ids, columns=['NLDAS_ID'])
temp_df = pd.merge(temp_df, max_land_constr, on='NLDAS_ID',how='left')
max_land_constr_dict = temp_df['max_land_constr'].to_dict()
with open('net_prices_new.p', 'wb') as handle:
    pickle.dump(net_prices, handle, protocol=pickle.HIGHEST_PROTOCOL)

list_ids = [16173,70008,123843,177678,231513,285348,339183,393018,446853,500688]
for i in list_ids:
    print(net_prices[i])

with open('net_prices_corn2x.p', 'wb') as handle:
    pickle.dump(net_prices, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load in ABM csv results for cropped areas
water20 = pd.read_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_pmp\\local_debug\\run_output\\abm_results_20perclesswater.csv')
water20 = water20[(water20.crop=='Wheat')]
water20_pivot = pd.pivot_table(water20, index=['nldas'], values=['calc_area'],
                               aggfunc=np.sum)  # JY demand is order of magnitude low, double check calcs
water20_pivot.to_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_pmp\\local_debug\\run_output\\GIS_20water_wheat.csv')

water20 = pd.read_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_pmp\\local_debug\\run_output\\abm_results_baseline.csv')
water20 = water20[(water20.crop=='Wheat')]
water20_pivot = pd.pivot_table(water20, index=['nldas'], values=['calc_area'],
                               aggfunc=np.sum)  # JY demand is order of magnitude low, double check calcs
water20_pivot.to_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_pmp\\local_debug\\run_output\\GIS_baseline_wheat.csv')

with open('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_pmp\\data_inputs\\gammas_new_20201006.p', 'rb') as fp:
    gammas_old = pickle.load(fp, encoding='latin1')

with open('alpha_new_20201013_protocol2.p', 'wb') as handle:
    pickle.dump(alphas, handle, protocol=2)

ds = xr.open_dataset('C:\\Users\\yoon644\\Desktop\\usgs.mosart.h0.2000-08.nc')
df = ds.to_dataframe()
df = df['WRM_DEMAND0']

ds2 = xr.open_dataset('C:\\Users\\yoon644\\Desktop\\StatMod_consumption_2000_08.nc')
df2 = ds2.to_dataframe()

ds3 = xr.open_dataset('C:\\Users\\yoon644\\Desktop\\USGS_total_consumption_1999_08.nc')
df3 = ds3.to_dataframe()
df3 = df3['WRM_DEMAND0']

ds4 = xr.open_dataset('C:\\Users\\yoon644\\Desktop\\USGS_total_withdraw_1999_08.nc')
df4 = ds4.to_dataframe()

ds5 = xr.open_dataset('C:\\Users\\yoon644\\Desktop\\USGS_irr_withdraw_1999_08.nc')
df5 = ds5.to_dataframe()

ds6 = xr.open_dataset('C:\\Users\\yoon644\\Desktop\\USGS_irr_consumption_1999_08.nc')
df6 = ds6.to_dataframe()

mu = 0.2
hist_avail_bias = pd.read_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_pmp\\data_inputs\\hist_avail_bias_correction.csv')
hist_avail_bias['WRM_SUPPLY_acreft_prev'] = hist_avail_bias['WRM_SUPPLY_acreft_OG']
abm_supply_avail = pd.merge(abm_supply_avail, hist_avail_bias[['NLDAS_ID','sw_avail_bias_corr','WRM_SUPPLY_acreft_OG','WRM_SUPPLY_acreft_prev']], on=['NLDAS_ID'])
abm_supply_avail['WRM_SUPPLY_acreft_updated'] = ((1 - mu) * abm_supply_avail['WRM_SUPPLY_acreft_prev']) + (mu * abm_supply_avail['WRM_SUPPLY_acreft'])


##############################

# read a sample water demand input file
file = 'C:\\Users\\yoon644\\Desktop\\usgs.mosart.h0.2000-08.nc'
with netCDF4.Dataset(file, 'r') as nc:
    # for key, var in nc.variables.items():
    #     print(key, var.dimensions, var.shape, var.units, var.long_name, var._FillValue)

    lat = nc['lat'][:]
    lon = nc['lon'][:]
    demand = nc['WRM_DEMAND0'][:]

# read NLDAS grid reference file
df_grid = pd.read_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_pmp\\local_debug\\pmp_input_files_PIC_copy\\NLDAS_Grid_Reference.csv')

df_grid = df_grid[['CENTERX', 'CENTERY', 'NLDAS_X', 'NLDAS_Y', 'NLDAS_ID']]

df_grid = df_grid.rename(columns={"CENTERX": "longitude", "CENTERY": "latitude"})
df_grid['longitude'] = df_grid.longitude + 360

# match netCDF demand file and datagrame
mesh_lon, mesh_lat = np.meshgrid(lon, lat)
df_nc = pd.DataFrame({'lon':mesh_lon.reshape(-1,order='C'),'lat':mesh_lat.reshape(-1,order='C')})
df_nc['NLDAS_ID'] = ['x'+str(int((row['lon']-235.0625)/0.125+1))+'y'+str(int((row['lat']-25.0625)/0.125+1)) for _,row in df_nc.iterrows()]
df_nc['WRM_DEMAND0'] = 0

# use NLDAS_ID as index for both dataframes
df_nc = df_nc.set_index('NLDAS_ID',drop=False)
try:
    results_pivot = results_pivot.set_index('nldas',drop=False)
except KeyError:
    pass

# read ABM values into df_nc basing on the same index
df_nc.loc[results_pivot.index,'totalDemand'] = results_pivot.calc_water_demand.values

with open('net_prices_new_20201102.p', 'wb') as handle:
    pickle.dump(net_prices, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('max_land_constr_20201102_protocol2.p', 'wb') as handle:
    pickle.dump(land_constraints_by_farm, handle, protocol=2)

#############################
import pandas as pd
import xarray as xr
import pickle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

nldas = pd.read_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_netcdf\\nldas.txt')

with open('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_pmp\\local_debug\\pmp_input_files_PIC_copy\\nldas_ids.p', 'rb') as fp:
    nldas_ids = pickle.load(fp)

ds = xr.open_dataset('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\IM3\\WM Flag Tests\\US_reservoir_8th_NLDAS3_updated_CERF_Livneh_naturalflow.nc')
dams = ds["DamInd_2d"].to_dataframe()
dams = dams.reset_index()

dep = ds["gridID_from_Dam"].to_dataframe()
dep = dep.reset_index()
dep_id = ds["unit_ID"].to_dataframe()

dep_merge = pd.merge(dep, dams, how='left', left_on=['Dams'], right_on=['DamInd_2d'])

ds = xr.open_dataset('C:\\Users\\yoon644\\Desktop\\wm_abm_run.mosart.h0.2000-01.nc')
wm_results = ds[["WRM_STORAGE","WRM_DEMAND0","GINDEX","RIVER_DISCHARGE_OVER_LAND_LIQ"]].to_dataframe()
wm_results = wm_results.reset_index()
wm_results = pd.merge(wm_results, nldas, how='left', left_on=['lat', 'lon'], right_on=['CENTERY', 'CENTERX'])
wm_results = wm_results[wm_results['NLDAS_ID'].isin(nldas_ids)].reset_index()

dep_merge = pd.merge(dep_merge, wm_results[['lat','lon','WRM_STORAGE']], how='left', left_on=['lat','lon'], right_on=['lat','lon'])

aggregation_functions = {'WRM_STORAGE': 'sum'}
dep_merge = dep_merge.groupby(['gridID_from_Dam'], as_index=False, dropna=False).aggregate(aggregation_functions)
dep_merge.rename(columns={'WRM_STORAGE': 'STORAGE_SUM'}, inplace=True)

wm_results = pd.merge(wm_results, dep_merge, how='left', left_on=['GINDEX'], right_on=['gridID_from_Dam'])
abm_supply_avail = wm_results[wm_results['NLDAS_ID'].isin(nldas_ids)].reset_index()
abm_supply_avail = abm_supply_avail[['NLDAS_ID','STORAGE_SUM']]
abm_supply_avail = abm_supply_avail.fillna(0)

#################

data = pd.read_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\JWP\\Figures\\alldata_hh_vulnerability_max_40.csv')
data_subset = data[(data.intervention_name=='baseline') | (data.intervention_name=='demand management')]
data_subset = data_subset[(data_subset.scenario_name=='growth') | (data_subset.scenario_name=='crisis')]
data_subset = data_subset[(data_subset.year >= 2050)]
#data_subset = data_subset[['name','date','year','month','income','represented_units','hnum','conseq_months_below_40_lcd','population','scenario_name','intervention_name']]
data_subset = data_subset[['name','date','year','month','income','represented_units','hnum','piped','population','scenario_name','intervention_name']]
data_subset = data_subset[['name','year','month','represented_units','hnum','piped','scenario_name','intervention_name']]

import pandas as pd
import numpy as np

### Example shortage duration calc (currently post-processed in Tableau) for year 2099, demand management, population growth, low-income category

# load in source data
data_subset = pd.read_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\JWP\\Figures\\data_for_CK.csv')

# subset data
data_example = data_subset[(data_subset.year == 2099)]
data_example = data_example[(data_example.scenario_name == 'growth')]
data_example = data_example[(data_example.intervention_name == 'demand management')]
data_example = data_example[(data_example.income < 2412.3711340206196)] # cutoff between low and high income groups

# revised shortage duration calc
data_example['conseq_months_max_12'] = np.where(data_example['conseq_months_below_40_lcd'] > 12.0, 12.0, data_example['conseq_months_below_40_lcd']) # set the max consecutive months of shortage to 12
data_example['conseq_months_max_12'] = data_example['conseq_months_below_40_lcd']

# weight shortage duration calc by population for each agent
data_example['weighted_shortage'] = data_example['conseq_months_max_12'] * data_example['population']

# calculate avg population weighted shortage duration for entire subset
weighted_shortage_calc = data_example.weighted_shortage.sum() / data_example.population.sum()

# print result
print(weighted_shortage_calc)