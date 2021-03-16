import pandas as pd
import os
import xarray as xr
import numpy as np
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# switch to directory with ABM runs
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\ABM runs v8')

# Load in NLDAS/HUC-2 join table
huc2 = pd.read_csv('NLDAS_HUC2_join.csv')

# Load in states/counties/regions join table
states_etc = pd.read_csv('nldas_states_counties_regions.csv')

# Develop csv file for visualizing crop areas in Tableau
# Load in ABM csv results for cropped areas
for year in range(70):
    abm = pd.read_csv('abm_results_' + str(year+1940))
    aggregation_functions = {'calc_area': 'sum'}
    if year == 0:
        abm = pd.merge(abm, huc2[['NLDAS_ID', 'NAME']], how='left',left_on='nldas',right_on='NLDAS_ID')
        abm = pd.merge(abm, states_etc[['COUNTYFP','ERS_region','State','NLDAS_ID']],how='left',left_on='nldas',right_on='NLDAS_ID')
        abm_detailed = abm
        abm_detailed['year'] = year+1940
        abm_summary = abm.groupby(['crop','NAME'], as_index=False).aggregate(aggregation_functions)
        abm_summary['year'] = year+1940
    else:
        abm = pd.merge(abm, huc2[['NLDAS_ID', 'NAME']], how='left',left_on='nldas',right_on='NLDAS_ID')
        abm = pd.merge(abm, states_etc[['COUNTYFP','ERS_region','State','NLDAS_ID']],how='left',left_on='nldas',right_on='NLDAS_ID')
        abm['year'] = year + 1940
        abm_detailed = abm_detailed.append(abm)
        abm_summary_to_append = abm.groupby(['crop','NAME'], as_index=False).aggregate(aggregation_functions)
        abm_summary_to_append['year'] = year+1940
        abm_summary = abm_summary.append(abm_summary_to_append)

# Join to sigmoid model
abm_summary['Join'] = 1
sigmoid = pd.read_csv('AreaBumpModelv3.csv')
abm_summary = pd.merge(abm_summary, sigmoid, on='Join', how='inner')
abm_summary = abm_summary.rename(columns={"crop": "Sub-category", "NAME": "_Category", "calc_area": "Total", "year": "Year"})
abm_summary.to_csv('abm_join_v7_sigmoidv3.csv', index=False)

# Join wm_results_summary file (processed on PIC via wm_pmp/WM_output_PIC_ncdf.py) with various geographies
wm_results = pd.read_csv('wm_summary_results.csv')
wm_results = pd.merge(wm_results, huc2[['NLDAS_ID','NAME']], how='left', on='NLDAS_ID')
wm_results = pd.merge(wm_results, states_etc[['NLDAS_ID','ERS_region','State']], how='left',on='NLDAS_ID')
wm_results.to_csv('wm_summary_results_join_v7.csv')

# Read wm_results_summary file for baseline (no abm) and merge with above
wm_results['experiment'] = 'abm'
wm_results_noabm = pd.read_csv('wm_summary_results_noabm.csv')
wm_results_noabm = pd.merge(wm_results_noabm, huc2[['NLDAS_ID','NAME']], how='left', on='NLDAS_ID')
wm_results_noabm = pd.merge(wm_results_noabm, states_etc[['NLDAS_ID','ERS_region','State']], how='left',on='NLDAS_ID')
wm_results_noabm['experiment'] = 'no abm'
wm_results = wm_results.append(wm_results_noabm)
wm_results.to_csv('wm_results_noabm_compare.csv')

# Calculate U.S. average abm/baseline demand
df = pd.read_csv('wm_results_noabm_compare.csv')
df_abm = df[(df.experiment=='abm')]
df_base = df[(df.experiment=='no abm')]
aggregation_functions = {'WRM_DEMAND0': 'sum'}
df_abm = df_abm.groupby(['NLDAS'], as_index=False).aggregate(aggregation_functions)
df_abm = df_abm.rename(columns={'WRM_DEMAND0': 'abm_demand'})
df_base = df_base.groupby(['year'], as_index=False).aggregate(aggregation_functions)
df_base = df_base.rename(columns={'WRM_DEMAND0': 'base_demand'})
df_abm = pd.merge(df_abm, df_base, how='left',on='year')
df_abm['abm/baseline demand'] = df_abm['abm_demand'] / df_abm['base_demand']
df_abm.to_csv('abm_div_base_demand.csv')

# Calculate NLDAS average abm/baseline demand (for visualization in QGIS)
df = pd.read_csv('wm_results_noabm_compare.csv')
df = df[(df.year >= 1940)]
df_abm = df[(df.experiment == 'abm')]
df_base = df[(df.experiment == 'no abm')]
aggregation_functions = {'WRM_DEMAND0': 'sum'}
df_abm = df_abm.groupby(['NLDAS_ID'], as_index=False).aggregate(aggregation_functions)
df_abm = df_abm.rename(columns={'WRM_DEMAND0': 'abm_demand'})
df_base = df_base.groupby(['NLDAS_ID'], as_index=False).aggregate(aggregation_functions)
df_base = df_base.rename(columns={'WRM_DEMAND0': 'base_demand'})
df_abm = pd.merge(df_abm, df_base, how='left', on='NLDAS_ID')
df_abm['a_div_b'] = df_abm['abm_demand'] / df_abm['base_demand']
df_abm = df_abm.fillna(0)
df_abm = df_abm.replace([np.inf, -np.inf], 0)
df_abm.to_csv('abm_div_base_demand_GIS.csv')

# Calculate the max level of shortage per NLDAS grid cell
wm_results['shortage_perc'] = 1.0 - (wm_results['WRM_SUPPLY'] / wm_results['WRM_DEMAND0'])
wm_results = wm_results[(wm_results.year>=1950)]
wm_results = wm_results.fillna(0)
wm_results_max_shortage = wm_results.loc[wm_results.groupby("NLDAS_ID")["shortage_perc"].idxmax()]

# Calculate the avg demand and take difference between abm run and no abm run
wm_results = wm_results[(wm_results.year>=1950)]
wm_results = wm_results.fillna(0)
#wm_results_max = wm_results.loc[wm_results.groupby("NLDAS_ID")["WRM_DEMAND0"].idxmax()]
aggregation_functions = {'WRM_DEMAND0': 'mean','NAME': 'first', 'ERS_region': 'first', 'State': 'first'}
wm_results_avg = wm_results.groupby(['NLDAS_ID'], as_index=False).aggregate(aggregation_functions)
wm_results_avg = wm_results_avg.rename(columns={"WRM_DEMAND0": "WRM_DEMAND0_abm"})
wm_results_noabm = wm_results_noabm[(wm_results_noabm.year>=1950)]
wm_results_noabm = wm_results_noabm.fillna(0)
#wm_results_max_noabm = wm_results_noabm.loc[wm_results_noabm.groupby("NLDAS_ID")["WRM_DEMAND0"].idxmax()]
aggregation_functions = {'WRM_DEMAND0': 'mean','NAME': 'first', 'ERS_region': 'first', 'State': 'first'}
wm_results_avg_noabm = wm_results_noabm.groupby(['NLDAS_ID'], as_index=False).aggregate(aggregation_functions)
wm_results_avg_noabm = wm_results_avg_noabm.rename(columns={"WRM_DEMAND0": "WRM_DEMAND0_noabm"})
wm_results_difference = pd.merge(wm_results_avg[['NLDAS_ID','NAME','ERS_region','State','WRM_DEMAND0_abm']], wm_results_avg_noabm[['NLDAS_ID','WRM_DEMAND0_noabm']], how='left', on='NLDAS_ID')
wm_results_difference['avg_demand_diff'] = wm_results_difference['WRM_DEMAND0_abm'] - wm_results_difference['WRM_DEMAND0_noabm']

# Calculate the max level of shortage and take difference between abm run and no abm run
wm_results['shortage_perc_abm'] = 1.0 - (wm_results['WRM_SUPPLY'] / wm_results['WRM_DEMAND0'])
wm_results = wm_results[(wm_results.year>=1950)]
wm_results = wm_results.fillna(0)
wm_results_max = wm_results.loc[wm_results.groupby("NLDAS_ID")["shortage_perc_abm"].idxmax()]
wm_results_noabm['shortage_perc_noabm'] = 1.0 - (wm_results_noabm['WRM_SUPPLY'] / wm_results_noabm['WRM_DEMAND0'])
wm_results_noabm = wm_results_noabm[(wm_results_noabm.year>=1950)]
wm_results_noabm = wm_results_noabm.fillna(0)
wm_results_max_noabm = wm_results_noabm.loc[wm_results_noabm.groupby("NLDAS_ID")["shortage_perc_noabm"].idxmax()]
wm_results_difference = pd.merge(wm_results_max[['NLDAS_ID','NAME','ERS_region','State','shortage_perc_abm']], wm_results_max_noabm[['NLDAS_ID','shortage_perc_noabm']], how='left', on='NLDAS_ID')
wm_results_difference['shortage_diff'] = wm_results_difference['shortage_perc_noabm'] - wm_results_difference['shortage_perc_abm']

# Calculate minimum river flow difference between abm run and no abm run
wm_results = wm_results[(wm_results.year>=1955)]
wm_results = wm_results.fillna(0)
wm_results_river_min = wm_results.loc[wm_results.groupby("NLDAS_ID")["RIVER_DISCHARGE_OVER_LAND_LIQ"].idxmin()]
wm_results_river_min = wm_results_river_min.rename(columns={"RIVER_DISCHARGE_OVER_LAND_LIQ": "river_min_abm"})
wm_results_noabm = wm_results_noabm[(wm_results_noabm.year>=1955)]
wm_results_noabm = wm_results_noabm.fillna(0)
wm_results_river_min_noabm = wm_results_noabm.loc[wm_results_noabm.groupby("NLDAS_ID")["RIVER_DISCHARGE_OVER_LAND_LIQ"].idxmin()]
wm_results_river_min_noabm = wm_results_river_min_noabm.rename(columns={"RIVER_DISCHARGE_OVER_LAND_LIQ": "river_min_noabm"})
wm_results_river_difference = pd.merge(wm_results_river_min[['NLDAS_ID','NAME','ERS_region','State','river_min_abm']], wm_results_river_min_noabm[['NLDAS_ID','river_min_noabm']], how='left', on='NLDAS_ID')
wm_results_river_difference['perc_diff_min_flow'] = (wm_results_river_difference['river_min_abm'] - wm_results_river_difference['river_min_noabm'])/ wm_results_river_difference['river_min_noabm']

# Most prominent crop
abm = pd.read_csv('abm_results_1993')
abm = abm.loc[abm.groupby("nldas")["calc_area"].idxmax()]
abm[['nldas','crop','calc_area']].to_csv('crop_max_1993.csv')

# Determine number of cells that have switched prominent crop between two given years
abm_1993 = pd.read_csv('abm_results_1993')
abm_1993 = abm_1993.loc[abm_1993.groupby("nldas")["calc_area"].idxmax()]
abm_1985 = pd.read_csv('abm_results_1985')
abm_1985 = abm_1985.loc[abm_1985.groupby("nldas")["calc_area"].idxmax()]
abm_1993 = abm_1993.rename(columns={"crop": "1993_crop"})
abm_1985 = abm_1985.rename(columns={"crop": "1985_crop"})
abm_change = pd.merge(abm_1985[['nldas','1985_crop']], abm_1993[['nldas','1993_crop']], how='left',on='nldas')
abm_change['change_full'] = np.where(abm_change['1985_crop'] != abm_change['1993_crop'],abm_change['1985_crop'] + '_' + abm_change['1993_crop'],'No Change')
abm_change['change_short'] = np.where(abm_change['1985_crop'] != abm_change['1993_crop'],abm_change['1993_crop'],'No Change')
abm_change[(abm_change['change_short']=='No Change')].__len__()

# Working section

abm2000_max = abm2000.loc[abm2000.groupby("nldas")["calc_area"].idxmax()]

aggregation_functions = {'calc_area': 'sum', 'crop':''}

# Output crop-specific csv for joining with shapefile in QGIS
test2[(test2.crop=='Corn')].to_csv('corn_2000.csv')
test2[(test2.crop=='Corn')].to_csv('corn_2004.csv')

# Load in ABM csv results for cropped areas
nldas = pd.read_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\PyProjects\\wm_netcdf\\nldas.txt')

ds = xr.open_dataset('VIC_livneh_upperCO.nc')
df = ds.to_dataframe()
df = df.reset_index()

aggregation_functions = {'QRUNOFF': 'sum'}
df_agg = df.groupby(['time'], as_index=False).aggregate(aggregation_functions)

df_agg['time'] = pd.to_datetime(df['time'].astype(str))
df_agg['year'] = df['time'].dt.year
df_agg['month'] = df['time'].dt.month

df_agg.set_index('time', inplace=True)
df_agg.index = pd.to_datetime(df_agg.index)
df_agg = df_agg.resample('1M').mean()
df_agg.to_csv('upperCO_VIC_livneh_box.csv')
