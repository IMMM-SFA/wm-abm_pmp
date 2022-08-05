# This is a script for post-processing WM-ABM output files and generating a csv that in turn is post-processed
# in Tableau. Tableau is used to generate Figure XX.

import pandas as pd
import os
import pickle
import xarray as xr
import numpy as np
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# switch to directory with ABM runs
# os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202104 Mem 02 Corr')
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\IM3\\Paper #1\\Nature Communications submission\\Revision\\results')

# Load in NLDAS/HUC-2 join table
huc2 = pd.read_csv('NLDAS_HUC2_join.csv')

# Load in states/counties/regions join table
states_etc = pd.read_csv('nldas_states_counties_regions.csv')

# switch to directory with ABM runs
# os.chdir('C:\\Users\\yoon644\\Desktop\\corrected test')
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\IM3\\Paper #1\\Nature Communications submission\\Revision\\results\\20220420 abm')

# read in PMP input files for profit calc
pmp = pd.read_excel('MOSART_WM_PMP_inputs_20220323_GW.xlsx')
with open('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\IM3\\Paper #1\\Nature Communications submission\\Revision\\results\\20220420 abm\\gammas_total_dict_20220408_protocol2.p', 'rb') as fp:
    gammas_total = pickle.load(fp)
with open('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\IM3\\Paper #1\\Nature Communications submission\\Revision\\results\\20220420 abm\\net_prices_total_dict_20220408_protocol2.p', 'rb') as fp:
    net_prices_total = pickle.load(fp)
with open('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\IM3\\Paper #1\\Nature Communications submission\\Revision\\results\\20220420 abm\\alphas_total_dict_20220408_protocol2.p', 'rb') as fp:
    alphas_total = pickle.load(fp)
with open('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\IM3\\Paper #1\\Nature Communications submission\\Revision\\results\\20220420 abm\\net_prices_sw_20220323_protocol2.p', 'rb') as fp:
    net_prices_sw = pickle.load(fp)
with open('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\IM3\\Paper #1\\Nature Communications submission\\Revision\\results\\20220420 abm\\net_prices_gw_20220323_protocol2.p', 'rb') as fp:
    net_prices_gw = pickle.load(fp)

pmp['dict_map'] = pmp['index'] - 1
pmp['gammas_total'] = pmp.dict_map.map(gammas_total)
pmp['net_prices_total'] = pmp.dict_map.map(net_prices_total)
pmp['alphas_total'] = pmp.dict_map.map(alphas_total)
pmp['net_prices_sw'] = pmp.dict_map.map(net_prices_sw)
pmp['net_prices_gw'] = pmp.dict_map.map(net_prices_gw)

# Load abm results
for year in range(60):
    print(year)
    abm = pd.read_csv('abm_results_' + str(year+1950))
    aggregation_functions = {'profit_calc': 'sum', 'xs_gw': 'sum', 'xs_sw': 'sum', 'xs_total': 'sum'}
    if year == 0:
        abm_pmp = pd.merge(abm[['xs_gw','xs_sw','xs_total']], pmp, left_index=True, right_index=True)
        abm_pmp['profit_calc'] = (abm_pmp['net_prices_total'] * abm_pmp['xs_total']) - (0.5 * abm_pmp['gammas_total'] * abm_pmp['xs_total'] * abm_pmp['xs_total']) + \
                         (abm_pmp['net_prices_gw'] * abm_pmp['xs_gw']) + (abm_pmp['net_prices_sw'] * abm_pmp['xs_sw'])
        abm_pmp_summary = abm_pmp.groupby(['nldas'], as_index=False).aggregate(aggregation_functions)
        abm_pmp_summary['year'] = year+1950
    else:
        abm_pmp = pd.merge(abm[['xs_gw','xs_sw','xs_total']], pmp, left_index=True, right_index=True)
        abm_pmp['profit_calc'] = (abm_pmp['net_prices_total'] * abm_pmp['xs_total']) - (0.5 * abm_pmp['gammas_total'] * abm_pmp['xs_total'] * abm_pmp['xs_total']) + \
                         (abm_pmp['net_prices_gw'] * abm_pmp['xs_gw']) + (abm_pmp['net_prices_sw'] * abm_pmp['xs_sw'])
        abm_pmp_summary_to_append = abm_pmp.groupby(['nldas'], as_index=False).aggregate(aggregation_functions)
        abm_pmp_summary_to_append['year'] = year+1950
        abm_pmp_summary = abm_pmp_summary.append(abm_pmp_summary_to_append)

abm_pmp_summary = pd.merge(abm_pmp_summary, huc2[['NLDAS_ID', 'NAME']], how='left',left_on='nldas',right_on='NLDAS_ID')
abm_pmp_summary['profit_calc'] = abm_pmp_summary['profit_calc'] / 1000