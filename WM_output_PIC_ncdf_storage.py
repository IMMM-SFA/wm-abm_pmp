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

# Extract storage outputs from PIC (for Nature Communications second revision, 1/25/2023)
for year in range(70):
    print(str(year))
    year_str = str(year+1940)
    # ds = xr.open_dataset('/pic/scratch/yoon644/csmruns/wm_abm_run/run/abmrun_20230122/wm_abm_run.mosart.h0.' + year_str + '-' + '09' + '.nc')
    ds = xr.open_dataset('/pic/scratch/yoon644/csmruns/wm_abm_run/run/baselinerun_20230117/wm_abm_run.mosart.h0.' + year_str + '-' + '07' + '.nc')
    df = ds.to_dataframe()
    df = df[['WRM_SUPPLY','WRM_DEMAND0','RIVER_DISCHARGE_OVER_LAND_LIQ','WRM_STORAGE','GINDEX']]
    df_merge = pd.merge(df, nldas, how='left', left_on=['lat', 'lon'], right_on=['CENTERY', 'CENTERX'])
    df_merge = pd.merge(df_merge, huc2, on='NLDAS_ID')
    df_merge = pd.merge(df_merge, states_etc, on='NLDAS_ID')
    df_merge = df_merge.dropna()
    df_merge = df_merge.drop_duplicates()
    df_merge = df_merge[['WRM_SUPPLY','WRM_DEMAND0','RIVER_DISCHARGE_OVER_LAND_LIQ','WRM_STORAGE','NLDAS_ID','NAME','State','ERS_region','COUNTYFP','GINDEX']]
    # if m == '01':
    #     df_append = df_merge
    # else:
    #     df_append = df_append.append(df_merge)
    if year == 0:
        df_summary = df_merge
        df_summary['year'] = year+1940
    else:
        df_merge['year'] = year+1940
        df_summary = df_summary.append(df_merge)

# df_summary[['year','NLDAS_ID','WRM_SUPPLY','WRM_DEMAND0','RIVER_DISCHARGE_OVER_LAND_LIQ','WRM_STORAGE','GINDEX']].to_csv('wm_storage_results_abmrun_20230122.csv')
df_summary[['year','NLDAS_ID','WRM_SUPPLY','WRM_DEMAND0','RIVER_DISCHARGE_OVER_LAND_LIQ','WRM_STORAGE','GINDEX']].to_csv('wm_july_results_baselinerun_20230117.csv')

#
# for year in range(21):
#     abm = pd.read_csv('/pic/scratch/yoon644/csmruns/wm_abm_run/run/abm_results_' + str(year+2000))
#     aggregation_functions = {'calc_area': 'sum'}
#     if year == 0:
#         abm = pd.merge(abm, huc2[['NLDAS_ID', 'NAME']], how='left',left_on='nldas',right_on='NLDAS_ID')
#         abm = pd.merge(abm, states_etc[['COUNTYFP','ERS_region','State','NLDAS_ID']],how='left',left_on='nldas',right_on='NLDAS_ID')
#         abm_detailed = abm
#         abm_detailed['year'] = year+2000
#         abm_summary = abm.groupby(['crop','NAME'], as_index=False).aggregate(aggregation_functions)
#         abm_summary['year'] = year+2000
#     else:
#         abm = pd.merge(abm, huc2[['NLDAS_ID', 'NAME']], how='left',left_on='nldas',right_on='NLDAS_ID')
#         abm = pd.merge(abm, states_etc[['COUNTYFP','ERS_region','State','NLDAS_ID']],how='left',left_on='nldas',right_on='NLDAS_ID')
#         abm['year'] = year + 2000
#         abm_detailed = abm_detailed.append(abm)
#         abm_summary_to_append = abm.groupby(['crop','NAME'], as_index=False).aggregate(aggregation_functions)
#         abm_summary_to_append['year'] = year+2000
#         abm_summary = abm_summary.append(abm_summary_to_append)
#
# abm_detailed.to_csv('abm_detailed_results.csv')
