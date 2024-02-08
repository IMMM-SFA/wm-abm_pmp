import pandas as pd
import xarray as xr
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

nldas = pd.read_csv('/pic/projects/im3/wm/Jim/wm_abm_postprocess/nldas.txt')

# Load in NLDAS/HUC-2 join table
huc2 = pd.read_csv('/pic/projects/im3/wm/Jim/wm_abm_postprocess/NLDAS_HUC2_join.csv')

# Load in states/counties/regions join table
states_etc = pd.read_csv('/pic/projects/im3/wm/Jim/wm_abm_postprocess/nldas_states_counties_regions.csv')

months = ['01','02','03','04','05','06','07','08','09','10','11','12']

for year in range(70):
    print(str(year))
    for m in months:
        year_str = str(year+1940)
        ds = xr.open_dataset('/pic/scratch/yoon644/csmruns/wm_abm_run/run/baselinerun_20230213/wm_abm_run.mosart.h0.' + year_str + '-' + m + '.nc')
        df = ds.to_dataframe()
        df = df[['WRM_SUPPLY','WRM_DEMAND0','RIVER_DISCHARGE_OVER_LAND_LIQ','WRM_STORAGE','GINDEX']]
        df_merge = pd.merge(df, nldas, how='left', left_on=['lat', 'lon'], right_on=['CENTERY', 'CENTERX'])
        df_merge = pd.merge(df_merge, huc2, on='NLDAS_ID')
        df_merge = pd.merge(df_merge, states_etc, on='NLDAS_ID')
        df_merge = df_merge.dropna()
        df_merge = df_merge.drop_duplicates()
        df_merge = df_merge[['WRM_SUPPLY','WRM_DEMAND0','RIVER_DISCHARGE_OVER_LAND_LIQ','WRM_STORAGE','NLDAS_ID','NAME','State','ERS_region','COUNTYFP','GINDEX']]
        df_merge['shortage'] = df_merge['WRM_DEMAND0'] - df_merge['WRM_SUPPLY']
        df_merge['year'] = year+1940
        df_merge['month'] = int(m)
        aggregation_functions = {'WRM_SUPPLY': 'sum', 'WRM_DEMAND0': 'sum', 'RIVER_DISCHARGE_OVER_LAND_LIQ': 'mean',
                                 'WRM_STORAGE': 'mean', 'shortage': 'sum', 'State': 'first', 'ERS_region': 'first',
                                 'COUNTYFP': 'first', 'GINDEX': 'first'}
        df_summary = df_merge.groupby(['NAME', 'year', 'month'], as_index=False).aggregate(aggregation_functions)
        if year == 0 and m == '01':
            df_append = df_summary
        else:
            df_append = df_append.append(df_summary)

df_append[['NAME','year','month','WRM_SUPPLY','WRM_DEMAND0','RIVER_DISCHARGE_OVER_LAND_LIQ','WRM_STORAGE','shortage', 'GINDEX']].to_csv('wm_summary_results_baselinerun_monthly_20230213.csv')
