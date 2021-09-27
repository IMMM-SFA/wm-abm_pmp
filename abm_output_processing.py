# This is a script that is used to post-process WM-ABM data and generate various csv files that are subsequently
# loaded into Tableau or QGIS for visualization and generation of paper figures.

import pandas as pd
import os
import xarray as xr
import numpy as np
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# switch to directory with ABM runs
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202104 Mem 02 Corr')

# Load in NLDAS/HUC-2 join table
huc2 = pd.read_csv('NLDAS_HUC2_join.csv')

# Load in states/counties/regions join table
states_etc = pd.read_csv('nldas_states_counties_regions.csv')

# Develop csv file for visualizing crop areas in Tableau
# Load in ABM csv results for cropped areas
for year in range(70):
    print(year)
    abm = pd.read_csv('abm_results_' + str(year+1940))
    #abm = abm[(abm.nldas=='x309y67')] ### JY TEMP
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
abm_summary['Year'] = abm_summary['Year'] - 1949  # shift years to start at 1
abm_summary['Total'] = abm_summary['Total'] / 1000  # correct areas to be in appropriate unit (acres)
abm_summary.to_csv('abm_join_202104_mem02_corr.csv', index=False)
#abm_summary.to_csv('abm_join_x309y67_sigmoid.csv', index=False)

# Join wm_results_summary file (processed on PIC via wm_pmp/WM_output_PIC_ncdf.py) with various geographies
wm_results = pd.read_csv('wm_summary_results_mem02_corr.csv')
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
wm_results.to_csv('wm_results_noabm_compare_mem02_corr.csv')

# Calculate U.S. average abm/baseline demand
df = pd.read_csv('wm_results_noabm_compare_mem02_corr.csv')
df_abm = df[(df.experiment=='abm')]
df_base = df[(df.experiment=='no abm')]
aggregation_functions = {'WRM_DEMAND0': 'sum'}
df_abm = df_abm.groupby(['year'], as_index=False).aggregate(aggregation_functions)
df_abm = df_abm.rename(columns={'WRM_DEMAND0': 'abm_demand'})
df_base = df_base.groupby(['year'], as_index=False).aggregate(aggregation_functions)
df_base = df_base.rename(columns={'WRM_DEMAND0': 'base_demand'})
df_abm = pd.merge(df_abm, df_base, how='left',on='year')
df_abm['abm/baseline demand'] = df_abm['abm_demand'] / df_abm['base_demand']
df_abm.to_csv('abm_div_base_demand_mem02_corr.csv')

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

# Calculate max shortage difference between abm and no abm (for visualization in QGIS)
df = pd.read_csv('wm_results_noabm_compare_mem02_corr.csv')
df = df[(df.year >= 1950)]
df['shortage'] = df['WRM_DEMAND0'] - df['WRM_SUPPLY']
df_abm = df[(df.experiment == 'abm')]
df_base = df[(df.experiment == 'no abm')]
aggregation_functions = {'shortage': 'max'}
df_abm = df_abm.groupby(['NLDAS_ID'], as_index=False).aggregate(aggregation_functions)
df_abm = df_abm.rename(columns={'shortage': 'abm_shortage'})
df_base = df_base.groupby(['NLDAS_ID'], as_index=False).aggregate(aggregation_functions)
df_base = df_base.rename(columns={'shortage': 'base_shortage'})
df_abm = pd.merge(df_abm, df_base, how='left', on='NLDAS_ID')
df_abm['shoratge_diff_abs'] = (df_abm['abm_shortage']) - (df_abm['base_shortage'])
df_abm.to_csv('abm_max_shortage_diff_abs_GIS.csv')

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

# Combine agent adaptivity classification and water shortage metrics
adapt = pd.read_csv('abm_output_classification_mem02_corr_v2.csv')
wm = pd.read_csv('wm_results_noabm_compare_mem02_corr.csv')
wm = wm[(wm.experiment=='abm')]
wm['shortage'] = (wm['WRM_DEMAND0'] - wm['WRM_SUPPLY']) / wm['WRM_DEMAND0']
aggregation_functions = {'shortage': 'max'}
wm_group = wm.groupby(['NLDAS_ID'], as_index=False).aggregate(aggregation_functions)
wm_group['shortage_class'] = np.where((wm_group['shortage'] >= 0.1), 'yes', 'no')
adapt = pd.merge(adapt, wm_group, how='left',left_on='nldas',right_on='NLDAS_ID')
adapt = adapt[(adapt.shortage_class == 'yes')]
adapt.to_csv('shortage_only_cells_10perc_thres.csv')

# Sensitivity shortage results normalized by baseline value and rank ordered
sens = pd.read_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202104 Mem 02 Corr\\abm_sensitivity_shortage_comp_perc.csv')
base = sens[(sens.sen_run=='base')]
base['rank'] = base.groupby("NAME")["shortage_perc"].rank("dense", ascending=False)
base = base[['year','NAME','shortage_perc','rank']]
base.rename(columns={'shortage_perc':'shortage_perc_base', 'rank':'rank_base'}, inplace=True)
sens = pd.merge(sens, base, how='left', on=['year','NAME'])
sens['shortage_diff_to_base'] = (sens['shortage_perc'] - sens['shortage_perc_base'])
sens['shortage_perc_diff_to_base'] = (sens['shortage_perc'] - sens['shortage_perc_base']) / sens['shortage_perc_base']
sens = sens.replace([np.inf, -np.inf], np.nan)
sens = sens.fillna(0)
sens.to_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202104 Mem 02 Corr\\abm_sensitivity_shortage_comp_perc_ranked.csv')

# Sensitivity shortage results (not normalized by baseline value) and rank ordered
sens = pd.read_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202104 Mem 02 Corr\\abm_sensitivity_shortage_comp_perc.csv')
base = sens[(sens.sen_run=='base')]
base['rank'] = base.groupby("NAME")["shortage_perc"].rank("dense", ascending=False)
base = base[['year','NAME','shortage_perc','rank']]
base.rename(columns={'shortage_perc':'shortage_perc_base', 'rank':'rank_base'}, inplace=True)
sens = pd.merge(sens, base, how='left', on=['year','NAME'])
sens['shortage_diff_to_base'] = (sens['shortage_perc'] - sens['shortage_perc_base'])
sens['shortage_perc_diff_to_base'] = (sens['shortage_perc'] - sens['shortage_perc_base']) / sens['shortage_perc_base']
sens = sens.replace([np.inf, -np.inf], np.nan)
sens = sens.fillna(0)
sens.to_csv('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202104 Mem 02 Corr\\abm_sensitivity_shortage_comp_perc_ranked.csv')

##### Working section

adapt = pd.merge(adapt, huc2[['NLDAS_ID','NAME']], how='left',on='NLDAS_ID')
adapt_subset = adapt[(adapt.classification == 'crop_switching') | (adapt.classification == 'crop_expansion') | (adapt.classification == 'both')]
adapt_subset = adapt_subset[(adapt_subset.NAME == 'Missouri Region')]
adapt_subset_NLDAS = adapt_subset.NLDAS_ID.unique().tolist()

sen_adapt = pd.merge(sen, adapt[['nldas','classification']], how='left',left_on='NLDAS_ID',right_on='nldas')


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

#### Apply classifier to designate land use change category ("steady", "crop expansion/contraction", "crop switching", "both")

for year in range(60):
    print(str(year))
    abm = pd.read_csv('abm_results_' + str(year+1950))

    # Create new dataframe to sum areas for each nldas cell
    aggregation_functions = {'calc_area': 'sum'}
    abm_sum_area = abm.groupby(['nldas'], as_index=False).aggregate(aggregation_functions)
    abm_sum_area = abm_sum_area.rename(columns={"calc_area": "sum_area"})

    if year == 0:
        # Subset original dataframe by max crop for each nldas cell
        abm_max_crop = abm.sort_values('calc_area', ascending=False).drop_duplicates(['nldas'])

        # Merge two dataframes
        abm_initial = pd.merge(abm_sum_area, abm_max_crop[['nldas', 'crop','calc_area']], how='left',on='nldas')
        abm_initial['max_crop_perc'] = abm_initial['calc_area'] / abm_initial['sum_area']
        abm_summary = abm_initial
        abm_summary['year'] = year + 1950
    else:
        abm_merge = pd.merge(abm_initial[['nldas', 'crop']], abm[['nldas', 'crop', 'calc_area']], how='left', on=['nldas', 'crop'])
        abm_merge = pd.merge(abm_sum_area, abm_merge[['nldas','crop','calc_area']])
        abm_merge['max_crop_perc'] = abm_merge['calc_area'] / abm_merge['sum_area']
        abm_merge['year'] = year + 1950
        abm_summary = abm_summary.append(abm_merge)

aggregation_functions = {'max_crop_perc': 'min'}
abm_crop_perc_min = abm_summary.groupby(['nldas','crop'], as_index=False).aggregate(aggregation_functions)
abm_crop_perc_min = abm_crop_perc_min.rename(columns={"max_crop_perc": "min_crop_perc"})

aggregation_functions = {'max_crop_perc': 'max'}
abm_crop_perc_max = abm_summary.groupby(['nldas','crop'], as_index=False).aggregate(aggregation_functions)
abm_crop_perc_max = abm_crop_perc_max.rename(columns={"max_crop_perc": "max_crop_perc"})

aggregation_functions = {'sum_area': 'min'}
abm_sum_area_min = abm_summary.groupby(['nldas','crop'], as_index=False).aggregate(aggregation_functions)
abm_sum_area_min = abm_sum_area_min.rename(columns={"sum_area": "min_sum_area"})

aggregation_functions = {'sum_area': 'max'}
abm_sum_area_max = abm_summary.groupby(['nldas','crop'], as_index=False).aggregate(aggregation_functions)
abm_sum_area_max = abm_sum_area_max.rename(columns={"sum_area": "max_sum_area"})

abm_min_max = pd.merge(abm_crop_perc_min, abm_crop_perc_max[['nldas','max_crop_perc']], how='left',on='nldas')
abm_min_max = pd.merge(abm_min_max, abm_sum_area_min[['nldas','min_sum_area']], how='left',on='nldas')
abm_min_max = pd.merge(abm_min_max, abm_sum_area_max[['nldas','max_sum_area']], how='left',on='nldas')

abm_min_max['classification'] = "none"
abm_min_max['classification'] = np.where((abm_min_max['max_crop_perc'] - abm_min_max['min_crop_perc'] > 0.2), 'crop switching', '0')
abm_min_max['classification'] = np.where((abm_min_max['min_sum_area'] / abm_min_max['max_sum_area'] < 0.8), 'crop expansion', '0')
abm_min_max['classification'] = np.where((abm_min_max['min_sum_area'] / abm_min_max['max_sum_area'] < 0.8) & (abm_min_max['max_crop_perc'] - abm_min_max['min_crop_perc'] > 0.2), 'both', '0')

abm_min_max.loc[(abm_min_max['max_crop_perc'] - abm_min_max['min_crop_perc'] > 0.2), 'classification'] = 'crop_switching'
abm_min_max.loc[(abm_min_max['min_sum_area'] / abm_min_max['max_sum_area'] < 0.8), 'classification'] = 'crop_expansion'
abm_min_max.loc[(abm_min_max['min_sum_area'] / abm_min_max['max_sum_area'] < 0.8) & (abm_min_max['max_crop_perc'] - abm_min_max['min_crop_perc'] > 0.2), 'classification'] = 'both'


