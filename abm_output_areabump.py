# This is a script for post-processing WM-ABM output files and generating a csv that in turn is post-processed
# in Tableau. Tableau is used to generate Figure 1a.

import pandas as pd
import os
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


# Develop csv file for visualizing crop areas in Tableau
# Load in ABM csv results for cropped areas
for year in range(60): # change back to 70
    print(year)
    abm = pd.read_csv('abm_results_' + str(year+1950))  # change back to 1940
    abm['nldas'] = abm['nldas'].astype(str)
    aggregation_functions = {'xs_gw': 'sum', 'xs_sw': 'sum', 'xs_total': 'sum'}
    # Redistribute crops for GW/SW model (otherwise crop allocation split between GW/SW at farm level is arbitrary)
    reallocate_df = abm.groupby(['nldas'], as_index=False).aggregate(aggregation_functions)
    reallocate_df['perc_gw'] = 0
    reallocate_df.loc[(reallocate_df.xs_total > 0), 'perc_gw'] = reallocate_df['xs_gw'] / reallocate_df['xs_total']
    reallocate_df['perc_sw'] = 0
    reallocate_df.loc[(reallocate_df.xs_total > 0), 'perc_sw'] = reallocate_df['xs_sw'] / reallocate_df['xs_total']
    abm = pd.merge(abm, reallocate_df[['nldas', 'perc_gw','perc_sw']], how='left',on='nldas')
    abm['xs_gw_reallo'] = abm['xs_total'] * abm['perc_gw']
    abm['xs_sw_reallo'] = abm['xs_total'] * abm['perc_sw']
    # aggregation_functions = {'calc_area': 'sum'}
    aggregation_functions = {'xs_sw_reallo': 'sum'}
    if year == 0:
        abm = pd.merge(abm, huc2[['NLDAS_ID', 'NAME']], how='left',left_on='nldas',right_on='NLDAS_ID')
        abm = pd.merge(abm, states_etc[['COUNTYFP','ERS_region','State','NLDAS_ID']],how='left',left_on='nldas',right_on='NLDAS_ID')
        abm_detailed = abm
        abm_detailed['year'] = year+1950  # change back to 1940
        abm_summary = abm.groupby(['crop','NAME'], as_index=False).aggregate(aggregation_functions)
        abm_summary['year'] = year+1950
    else:
        abm = pd.merge(abm, huc2[['NLDAS_ID', 'NAME']], how='left',left_on='nldas',right_on='NLDAS_ID')
        abm = pd.merge(abm, states_etc[['COUNTYFP','ERS_region','State','NLDAS_ID']],how='left',left_on='nldas',right_on='NLDAS_ID')
        abm['year'] = year + 1950
        #abm_detailed = abm_detailed.append(abm)
        abm_summary_to_append = abm.groupby(['crop','NAME'], as_index=False).aggregate(aggregation_functions)
        abm_summary_to_append['year'] = year+1950
        abm_summary = abm_summary.append(abm_summary_to_append)

# Join to sigmoid model
abm_summary['Join'] = 1
sigmoid = pd.read_csv('AreaBumpModelv3.csv')
abm_summary = pd.merge(abm_summary, sigmoid, on='Join', how='inner')
abm_summary = abm_summary.rename(columns={"crop": "Sub-category", "NAME": "_Category", "xs_sw_reallo": "Total", "year": "Year"})
# abm_summary = abm_summary.rename(columns={"crop": "Sub-category", "NAME": "_Category", "calc_area": "Total", "year": "Year"})
abm_summary['Total'] = abm_summary['Total'] / 1000  # correct areas to be in appropriate unit (acres)
abm_summary.to_csv('abm_join_sigmoid_NCrev.csv', index=False)