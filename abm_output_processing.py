import pandas as pd
import os
import xarray as xr
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

os.chdir('C:\\Users\\yoon644\\Desktop\\ABM runs 2050')

# Load in NLDAS/HUC-2 join table
huc2 = pd.read_csv('NLDAS_HUC2_join.csv')

# Load in states/counties/regions join table
states_etc = pd.read_csv('nldas_states_counties_regions.csv')

# Load in ABM csv results for cropped areas
abm = pd.read_csv('abm_results_2000')
abm['year'] = 2000

for year in range(40):
    abm = pd.read_csv('abm_results_' + str(year+2000))
    aggregation_functions = {'calc_area': 'sum'}
    if year == 0:
        abm = pd.merge(abm, huc2[['NLDAS_ID', 'NAME']], how='left',left_on='nldas',right_on='NLDAS_ID')
        abm = pd.merge(abm, states_etc[['COUNTYFP','ERS_region','State','NLDAS_ID']],how='left',left_on='nldas',right_on='NLDAS_ID')
        abm_detailed = abm
        abm_detailed['year'] = year+2000
        abm_summary = abm.groupby(['crop','NAME'], as_index=False).aggregate(aggregation_functions)
        abm_summary['year'] = year+2000
    else:
        abm = pd.merge(abm, huc2[['NLDAS_ID', 'NAME']], how='left',left_on='nldas',right_on='NLDAS_ID')
        abm = pd.merge(abm, states_etc[['COUNTYFP','ERS_region','State','NLDAS_ID']],how='left',left_on='nldas',right_on='NLDAS_ID')
        abm['year'] = year + 2000
        abm_detailed = abm_detailed.append(abm)
        abm_summary_to_append = abm.groupby(['crop','NAME'], as_index=False).aggregate(aggregation_functions)
        abm_summary_to_append['year'] = year+2000
        abm_summary = abm_summary.append(abm_summary_to_append)

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
