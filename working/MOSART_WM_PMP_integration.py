from pyomo.environ import *
from pyomo.opt import SolverFactory
import os
import pandas as pd
import numpy as np
import xarray as xr
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
import shutil
import netCDF4
import logging

logging.basicConfig(filename='../app.log', level=logging.INFO)

def calc_demand(year, month):
    if int(month) == 1:
        with open('../data_inputs/nldas_ids.p', 'rb') as fp:
            nldas_ids = pickle.load(fp)

        nldas = pd.read_csv('../data_inputs/nldas.txt')

        year_int = int(year)
        months = ['01','02','03','04','05','06','07','08','09','10','11','12']

        with open('../data_inputs/water_constraints_by_farm_v2.p', 'rb') as fp:
            water_constraints_by_farm = pickle.load(fp)
        water_constraints_by_farm = dict.fromkeys(water_constraints_by_farm, 9999999999)

        ## Read in Water Availability Files from MOSART-PMP
        if year_int==2000:
            pass
        else:

            #pic_output_dir = '/pic/scratch/yoon644/csmruns/jimtest2/run/'
            pic_input_dir = '../pic_input_test/'

            # loop through .nc files and extract data
            first = True
            for m in months:
                #dataset_name = 'jim_abm_integration.mosart.h0.' + str(year-1) + '-' + m + '.nc'
                dataset_name = 'statemod.mosart.h0.' + str(year_int - 1) + '-' + m + '.nc'
                ds = xr.open_dataset(pic_input_dir+dataset_name)
                df = ds.to_dataframe()
                df_merge = pd.merge(df, nldas, how='left', left_on=['lat', 'lon'], right_on=['CENTERY', 'CENTERX'])
                df_select = df_merge[['NLDAS_ID', 'WRM_DEMAND0', 'WRM_SUPPLY', 'WRM_DEFICIT']]
                df_select['year'] = year_int
                df_select['month'] = int(m)
                if first:
                    df_all = df_select
                    first = False
                else:
                    df_all = pd.concat([df_all, df_select])

            # calculate average across timesteps
            df_pivot = pd.pivot_table(df_all, index=['NLDAS_ID'], values=['WRM_SUPPLY'],
                                      aggfunc=np.mean)  # units will be average monthly (m3/s)
            df_pivot = df_pivot.reset_index()

            # convert units from m3/s to acre-ft/yr
            df_pivot['WRM_SUPPLY_acreft'] = df_pivot['WRM_SUPPLY'] * 60 * 60 * 24 * 30.42 * 12 / 1233.48
            abm_supply_avail = df_pivot[df_pivot['NLDAS_ID'].isin(nldas_ids)].reset_index()
            water_constraints_by_farm = abm_supply_avail['WRM_SUPPLY_acreft'].to_dict()

        ## Read in PMP calibration files
        data_file=pd.ExcelFile("../data_inputs/MOSART_WM_PMP_inputs_v1.xlsx")
        data_profit = data_file.parse("Profit")
        water_nirs=data_profit["nir_corrected"]
        nirs=dict(water_nirs)

        ## C.1. Preparing model indices and constraints:
        ids = range(592185) # total number of crop and nldas ID combinations
        farm_ids = range(53835) # total number of farm agents / nldas IDs
        with open('../data_inputs/crop_ids_by_farm.p', 'rb') as fp:
            crop_ids_by_farm = pickle.load(fp)
        with open('../data_inputs/crop_ids_by_farm_and_constraint.p', 'rb') as fp:
            crop_ids_by_farm_and_constraint = pickle.load(fp)
        with open('../data_inputs/land_constraints_by_farm.p', 'rb') as fp:
            land_constraints_by_farm = pickle.load(fp)

        # Load gammas and alphas
        with open('../data_inputs/gammas.p', 'rb') as fp:
            gammas = pickle.load(fp)
        with open('../data_inputs/net_prices.p', 'rb') as fp:
            net_prices = pickle.load(fp)

        x_start_values=dict(enumerate([0.0]*3))

        ## C.2. 2st stage: Quadratic model included in JWP model simulations
        ## C.2.a. Constructing model inputs:
        ##  (repetition to be safe - deepcopy does not work on PYOMO models)
        fwm_s = ConcreteModel()
        fwm_s.ids = Set(initialize=ids)
        fwm_s.farm_ids = Set(initialize=farm_ids)
        fwm_s.crop_ids_by_farm = Set(fwm_s.farm_ids, initialize=crop_ids_by_farm)
        fwm_s.crop_ids_by_farm_and_constraint = Set(fwm_s.farm_ids, initialize=crop_ids_by_farm_and_constraint)
        fwm_s.net_prices = Param(fwm_s.ids, initialize=net_prices, mutable=True)
        fwm_s.gammas = Param(fwm_s.ids, initialize=gammas, mutable=True)
        fwm_s.land_constraints = Param(fwm_s.farm_ids, initialize=land_constraints_by_farm, mutable=True)
        fwm_s.water_constraints = Param(fwm_s.farm_ids, initialize=water_constraints_by_farm, mutable=True) #JY here need to read calculate new water constraints
        fwm_s.xs = Var(fwm_s.ids, domain=NonNegativeReals, initialize=x_start_values)
        fwm_s.nirs = Param(fwm_s.ids, initialize=nirs, mutable=True)

        ## C.2.b. 2nd stage model: Constructing functions:
        def obj_fun(fwm_s):
            return 0.00001*sum(sum((fwm_s.net_prices[i] * fwm_s.xs[i] - 0.5 * fwm_s.gammas[i] * fwm_s.xs[i] * fwm_s.xs[i]) for i in fwm_s.crop_ids_by_farm[f]) for f in fwm_s.farm_ids)
        fwm_s.obj_f = Objective(rule=obj_fun, sense=maximize)


        def land_constraint(fwm_s, ff,):
            return sum(fwm_s.xs[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.land_constraints[ff]
        fwm_s.c1 = Constraint(fwm_s.farm_ids, rule=land_constraint)

        def water_constraint(fwm_s, ff):
            return sum(fwm_s.xs[i]*fwm_s.nirs[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.water_constraints[ff]
        fwm_s.c2 = Constraint(fwm_s.farm_ids, rule=water_constraint)

        ## C.2.c Creating and running the solver:
        opt = SolverFactory("ipopt", solver_io='nl')
        results = opt.solve(fwm_s, keepfiles=False, tee=True)
        print(results.solver.termination_condition)

        ## D.1. Storing main model outputs:
        result_xs = dict(fwm_s.xs.get_values())

        # JY store results into a pandas dataframe
        results_pd = data_profit
        results_pd = results_pd.assign(calc_area=result_xs.values())
        results_pd = results_pd.assign(nir=nirs.values())
        results_pd['calc_water_demand'] = results_pd['calc_area'] * results_pd['nir']
        results_pivot = pd.pivot_table(results_pd, index=['nldas'], values=['calc_water_demand'], aggfunc=np.sum) #JY demand is order of magnitude low, double check calcs

        # read a sample water demand input file
        file = '../data_inputs/RCP8.5_GCAM_water_demand_1980_01_copy.nc'
        with netCDF4.Dataset(file, 'r') as nc:
            # for key, var in nc.variables.items():
            #     print(key, var.dimensions, var.shape, var.units, var.long_name, var._FillValue)

            lat = nc['lat'][:]
            lon = nc['lon'][:]
            demand = nc['totalDemand'][:]

        # read NLDAS grid reference file
        df_grid = pd.read_csv('../data_inputs/NLDAS_Grid_Reference.csv')

        df_grid = df_grid[['CENTERX', 'CENTERY', 'NLDAS_X', 'NLDAS_Y', 'NLDAS_ID']]

        df_grid = df_grid.rename(columns={"CENTERX": "longitude", "CENTERY": "latitude"})
        df_grid['longitude'] = df_grid.longitude + 360

        # match netCDF demand file and datagrame
        mesh_lon, mesh_lat = np.meshgrid(lon, lat)
        df_nc = pd.DataFrame({'lon':mesh_lon.reshape(-1,order='C'),'lat':mesh_lat.reshape(-1,order='C')})
        df_nc['NLDAS_ID'] = ['x'+str(int((row['lon']-235.0625)/0.125+1))+'y'+str(int((row['lat']-25.0625)/0.125+1)) for _,row in df_nc.iterrows()]
        df_nc['totalDemand'] = 0

        # use NLDAS_ID as index for both dataframes
        df_nc = df_nc.set_index('NLDAS_ID',drop=False)
        results_pivot = results_pivot.set_index('nldas',drop=False)

        # read ABM values into df_nc basing on the same index
        df_nc.loc[results_pivot.index,'totalDemand'] = results_pivot.calc_water_demand.values

        for month in months:
            str_year = str(year_int)
            new_fname = 'pic_output_test/RCP8.5_GCAM_water_demand_'+ str_year + '_' + month + '.nc' # define ABM demand input directory
            shutil.copyfile(file, new_fname)
            demand_ABM = df_nc.totalDemand.values.reshape(len(lat),len(lon),order='C')
            with netCDF4.Dataset(new_fname,'a') as nc:
                nc['totalDemand'][:] = np.ma.masked_array(demand_ABM,mask=nc['totalDemand'][:].mask)
    else:
        pass