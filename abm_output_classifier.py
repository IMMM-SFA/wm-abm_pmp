# This is a script for post-processing WM-ABM output that classifies agents based on their level of adaptivity. The
# script generates a csv which is subsequently loaded into QGIS for generation of Figure 1b.

import pandas as pd
import os
import xarray as xr
import numpy as np
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# switch to directory with ABM runs
os.chdir('C:\\Users\\yoon644\\OneDrive - PNNL\\Documents\\wm abm data\\wm abm results\\ABM runs\\202104 Mem 02 corr')

# Load in NLDAS/HUC-2 join table
huc2 = pd.read_csv('NLDAS_HUC2_join.csv')

# Load in states/counties/regions join table
states_etc = pd.read_csv('nldas_states_counties_regions.csv')

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

abm_summary['sum_area'] = abm_summary['sum_area'] / 1000  # convert areas to standard acres

aggregation_functions = {'sum_area': 'mean'}
abm_sum_area_avg = abm_summary.groupby(['nldas','crop'], as_index=False).aggregate(aggregation_functions)
abm_sum_area_avg = abm_sum_area_avg.rename(columns={"sum_area": "sum_area_avg"})

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
abm_min_max = pd.merge(abm_min_max, abm_sum_area_avg[['nldas','sum_area_avg']], how='left',on='nldas')

abm_min_max['classification'] = "none"

# threshold for significant crop area average area of grid cell (acres)
threshold_acres = 800

abm_min_max.loc[(abm_min_max['max_crop_perc'] - abm_min_max['min_crop_perc'] > 0.05), 'classification'] = 'crop_switching'
abm_min_max.loc[(abm_min_max['min_sum_area'] / abm_min_max['max_sum_area'] < 0.8), 'classification'] = 'crop_expansion'
abm_min_max.loc[(abm_min_max['min_sum_area'] / abm_min_max['max_sum_area'] < 0.8) & (abm_min_max['max_crop_perc'] - abm_min_max['min_crop_perc'] > 0.2), 'classification'] = 'both'
abm_min_max.loc[(abm_min_max['sum_area_avg'] < threshold_acres), 'classification'] = 'none'


abm_min_max.to_csv('abm_output_classification_mem02.corr.csv')