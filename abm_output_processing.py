import pandas as pd
import os
import xarray as xr
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

os.chdir('C:\\Users\\yoon644\\Desktop')

# Load in ABM csv results for cropped areas
abm2000 = pd.read_csv('abm_results_2000.csv')

for year in range(6):
    abm = pd.read_csv('abm_results_' + str(year+2000) + '.csv')
    aggregation_functions = {'calc_area': 'sum'}
    abm_summary = abm.groupby(['crop'], as_index=False).aggregate(aggregation_functions)
    abm_summary.to_csv(str(year)+'.csv')

abm2000_max = abm2000.loc[abm2000.groupby("nldas")["calc_area"].idxmax()]

aggregation_functions = {'calc_area': 'sum', 'crop':''}
aggregation_functions = {'cal'}

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
