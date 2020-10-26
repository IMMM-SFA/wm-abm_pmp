import pandas as pd
import xarray as xr

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

nldas = pd.read_csv('/pic/projects/im3/wm/Jim/wm_abm_postprocess/nldas.txt')

# Load in NLDAS/HUC-2 join table
huc2 = pd.read_csv('/pic/projects/im3/wm/Jim/wm_abm_postprocess/NLDAS_HUC2_join.csv')

# Load in states/counties/regions join table
states_etc = pd.read_csv('/pic/projects/im3/wm/Jim/wm_abm_postprocess/nldas_states_counties_regions.csv')

months = ['01','02','03','04','05','06','07','08','09','10','11','12']

for year in range(40):
    for m in months:
        year_str = str(year+2000)
        ds = xr.open_dataset('/pic/scratch/yoon644/csmruns/wm_abm_run/run/wm_abm_run.mosart.h0.' + year_str + '-' + m + '.nc')
        df = ds.to_dataframe()
        df = df[['WRM_SUPPLY']]
        df_merge = pd.merge(df, nldas, how='left', left_on=['lat', 'lon'], right_on=['CENTERY', 'CENTERX'])
        df_merge = pd.merge(df_merge, huc2, on='NLDAS_ID')
        df_merge = pd.merge(df_merge, states_etc, on='NLDAS_ID')
        df_merge = df_merge.dropna()
        df_merge = df_merge.drop_duplicates()
        df_merge = df_merge[['WRM_SUPPLY','NLDAS_ID','NAME','State','ERS_region','COUNTYFP']]
        if m == '01':
            df_append = df_merge
        else:
            df_append = df_append.append(df_merge)
    if year == 0:
        aggregation_functions = {'WRM_SUPPLY': 'mean','NAME': 'first', 'State': 'first', 'ERS_region': 'first', 'COUNTYFP': 'first'}
        df_summary = df_append.groupby(['NLDAS_ID'], as_index=False).aggregate(aggregation_functions)
        df_summary['year'] = year+1980
    else:
        aggregation_functions = {'WRM_SUPPLY': 'mean','NAME': 'first', 'State': 'first', 'ERS_region': 'first', 'COUNTYFP': 'first'}
        df_summary_toappend = df_append.groupby(['NLDAS_ID'], as_index=False).aggregate(aggregation_functions)
        df_summary_toappend['year'] = year+1980
        df_summary = df_summary.append(df_summary_toappend)

df_summary[['year','NLDAS_ID','WRM_SUPPLY']].to_csv('wm_summary_results.csv')
