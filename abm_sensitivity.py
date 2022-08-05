# This is a script that post-processes output from WM-ABM memory decay sensitivity runs and generates csv files
# that are input to Tableau or QGIS for figure development

import pandas as pd
import os
import xarray as xr
import numpy as np
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#### Generate Agent Classification Comparison Maps ####
# switch to directory with ABM mem 02 run
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202104 Mem 02 corr')

# read in output classification file (as generated from abm_output_classifier.py)
mem02 = pd.read_csv('abm_output_classification_mem02.corr.csv')
mem02 = mem02[['nldas','classification']]
mem02 = mem02.rename(columns={"classification": "mem02_class"})

# switch to directory with ABM mem 04 run
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202105 Mem 04 corr')

# read in output classification file (as generated from abm_output_classifier.py)
mem04 = pd.read_csv('abm_output_classification_mem04_corr.csv')
mem04 = mem04[['nldas','classification']]
mem04 = mem04.rename(columns={"classification": "mem04_class"})

# merge two dataframes
mem_comp = pd.merge(mem02, mem04, how='left',on='nldas')
mem_comp['compare'] = np.where(mem_comp['mem02_class'] == mem_comp['mem04_class'], 'no change', mem_comp['mem02_class'] + ' to ' + mem_comp['mem04_class'])
mem_comp = mem_comp[['nldas','compare']]
mem_comp.to_csv('abm_mem02_mem04_class_compare.csv')
#### End ####

#### Generate Shortage Comparison Plots ####
# switch to directory with ABM mem 02 run
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202104 Mem 02 corr')
base = pd.read_csv('wm_summary_results_noabm.csv')
base['sen_run'] = 'base'
mem02 = pd.read_csv('wm_summary_results_mem02_corr.csv')
mem02['sen_run'] = 'mem02'
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202105 Mem 04 corr')
mem04 = pd.read_csv('wm_summary_results_mem04_corr.csv')
mem04['sen_run'] = 'mem04'
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202105 Mem 06 corr')
mem06 = pd.read_csv('wm_summary_results_mem06_corr.csv')
mem06['sen_run'] = 'mem06'
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202105 Mem 08 corr')
mem08 = pd.read_csv('wm_summary_results_mem08_corr.csv')
mem08['sen_run'] = 'mem08'
mem_comp = pd.concat([base, mem02, mem04, mem06, mem08], ignore_index=True)
mem_comp = mem_comp[(mem_comp.year >= 1950)]
mem_comp['year'] = mem_comp['year'] - 1949
#mem_comp['shortage_perc'] = (mem_comp['WRM_DEMAND0'] - mem_comp['WRM_SUPPLY']) / mem_comp['WRM_DEMAND0']

#### Generate Shortage Comparison Plots / NC Revision ####
# switch to directory with ABM mem 02 run
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\IM3\\Paper #1\\Nature Communications submission\\Revision\\results\\20220420 abm')
base = pd.read_csv('wm_summary_results_20220418_noabm.csv')
base['sen_run'] = 'base'
mem02 = pd.read_csv('wm_summary_results_20220418_abm.csv')
mem02['sen_run'] = 'mem02'
mem04 = pd.read_csv('wm_summary_results_20220717_sens04.csv')
mem04['sen_run'] = 'mem04'
mem06 = pd.read_csv('wm_summary_results_20220720_sens06.csv')
mem06['sen_run'] = 'mem06'
mem08 = pd.read_csv('wm_summary_results_20220722_sens08.csv')
mem08['sen_run'] = 'mem08'
mem_comp = pd.concat([base, mem02, mem04, mem06, mem08], ignore_index=True)
mem_comp = mem_comp[(mem_comp.year >= 1950)]
mem_comp['year'] = mem_comp['year'] - 1949
#mem_comp['shortage_perc'] = (mem_comp['WRM_DEMAND0'] - mem_comp['WRM_SUPPLY']) / mem_comp['WRM_DEMAND0']

# sum supplies and demands at the HUC-2 basin level
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\IM3\\Paper #1\\Nature Communications submission\\Revision\\results\\20220420 abm')
huc2 = pd.read_csv('NLDAS_HUC2_join.csv')
mem_comp = pd.merge(mem_comp, huc2[['NLDAS_ID','NAME']], how='left',on='NLDAS_ID')
aggregation_functions = {'WRM_SUPPLY': 'sum', 'WRM_DEMAND0': 'sum'}
mem_comp_summary = mem_comp.groupby(['year','NAME','sen_run'], as_index=False).aggregate(aggregation_functions)
mem_comp_summary['shortage_perc'] = (mem_comp_summary['WRM_DEMAND0'] - mem_comp_summary['WRM_SUPPLY']) / mem_comp_summary['WRM_DEMAND0']
mem_comp_summary.to_csv('abm_sensitivity_shortage_comp_perc_NCrev.csv')

# only grab cells with shortage > 0 for at least 1 sensitivity run for at least 1 year
aggregation_functions = {'shortage_perc': 'mean'}
mem_comp_subset = mem_comp.groupby(['NLDAS_ID','year'], as_index=False).aggregate(aggregation_functions)
mem_comp_subset = mem_comp_subset[(mem_comp_subset.shortage_perc > 0)]
mem_comp = mem_comp.loc[mem_comp['NLDAS_ID'].isin(mem_comp_subset['NLDAS_ID'])]

# merge sensitivity run results with HUC-2 regions
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202104 Mem 02 corr')
huc2 = pd.read_csv('NLDAS_HUC2_join.csv')
mem_comp = pd.merge(mem_comp, huc2[['NLDAS_ID','NAME']], how='left',on='NLDAS_ID')
aggregation_functions = {'shortage_perc': 'mean'}
mem_comp_summary = mem_comp.groupby(['year','sen_run','NAME'], as_index=False).aggregate(aggregation_functions)
mem_comp_summary.to_csv('abm_sensitivity_shortage_comp_region_perc.csv')
#### End ####

#### Generate Shortage Comparison Maps ####
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202104 Mem 02 corr')
mem02 = pd.read_csv('wm_summary_results_mem02_corr.csv')
mem02['sen_run'] = 'mem02'
mem02['shortage_mem02'] = mem02['WRM_DEMAND0'] - mem02['WRM_SUPPLY']
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202105 Mem 04 corr')
mem04 = pd.read_csv('wm_summary_results_mem04_corr.csv')
mem04['shortage_mem04'] = mem04['WRM_DEMAND0'] - mem04['WRM_SUPPLY']

aggregation_functions = {'shortage_mem02': 'sum'}
mem02 = mem02.groupby(['NLDAS_ID'], as_index=False).aggregate(aggregation_functions)

aggregation_functions = {'shortage_mem04': 'sum'}
mem04 = mem04.groupby(['NLDAS_ID'], as_index=False).aggregate(aggregation_functions)

mem_shortage_comp = pd.merge(mem02, mem04[['NLDAS_ID','shortage_mem04']], how='left',on='NLDAS_ID')
mem_shortage_comp['shortage_comp_perc'] = (mem_shortage_comp['shortage_mem04'] - mem_shortage_comp['shortage_mem02']) / mem_shortage_comp['shortage_mem02']
mem_shortage_comp['shortage_comp_perc'] = mem_shortage_comp['shortage_comp_perc'].replace(np.inf, np.nan)
mem_shortage_comp['shortage_comp_perc'] = mem_shortage_comp['shortage_comp_perc'].fillna(0)

mem_shortage_comp.to_csv('abm_sensitivity_shortage_comp_perc_GIS.csv')
