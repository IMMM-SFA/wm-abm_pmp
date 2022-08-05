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
import sys

logging.basicConfig(filename='/pic/scratch/yoon644/csmruns/wm_abm_run/run/app.log',level=logging.INFO)

logging.info('Successfully loaded all Python modules')

sys.stdout = open('/pic/scratch/yoon644/csmruns/wm_abm_run/run/python_stdout.log', 'w')
sys.stderr = open('/pic/scratch/yoon644/csmruns/wm_abm_run/run/python_stderr.log', 'w')

def calc_demand(year, month):

    import pyutilib.subprocess.GlobalData
    pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False
    
    logging.info(sys.version_info)
    logging.info(pd.__version__)
    try:
        year_int = int(year)
        if year_int == 1940:
            logging.info('Trying to run ABM calc for month, year: ' + month + ' ' + year)
            if int(month) == 1:
                logging.info('Entering month 1 calculations: ' + month)

                with open('/pic/projects/im3/wm/Jim/pmp_input_files/nldas_ids.p', 'rb') as fp:
                    nldas_ids = pickle.load(fp)

                nldas = pd.read_csv('/pic/projects/im3/wm/Jim/pmp_input_files/nldas.txt')

                #!!!JY


                year_int = int(year)
                months = ['01','02','03','04','05','06','07','08','09','10','11','12']

                with open('/pic/projects/im3/wm/Jim/pmp_input_files/water_constraints_by_farm_pyt278.p', 'rb') as fp:
                    water_constraints_by_farm = pickle.load(fp)
                # water_constraints_by_farm = pd.read_pickle('/pic/projects/im3/wm/Jim/pmp_input_files/water_constraints_by_farm_v2.p')
                water_constraints_by_farm = dict.fromkeys(water_constraints_by_farm, 9999999999)


                ## Read in Water Availability Files from MOSART-PMP
                if year_int==1940:  # If year is before 1950 (warm-up period), use the external baseline water demand files
                    sw_constraints_by_farm = pd.read_csv('/pic/projects/im3/wm/Jim/pmp_input_files/hist_avail_bias_correction_20220223.csv') # Use baseline water demand data for warmup period
                    sw_constraints_by_farm = sw_constraints_by_farm[['NLDAS_ID','sw_irrigation_vol']].reset_index()
                    sw_constraints_by_farm = sw_constraints_by_farm['sw_irrigation_vol'].to_dict()

                ## Read in PMP calibration files
                data_file=pd.ExcelFile("/pic/projects/im3/wm/Jim/pmp_input_files/MOSART_WM_PMP_inputs_20220223_GW.xlsx")
                data_profit = data_file.parse("Profit")
                water_nirs=data_profit["nir_corrected"]
                nirs=dict(water_nirs)

                logging.info('I have successfully loaded PMP calibration files for month, year: ' + month + ' ' + year)

                ## C.1. Preparing model indices and constraints:
                ids = range(538350) # total number of crop and nldas ID combinations
                farm_ids = range(53835) # total number of farm agents / nldas IDs
                with open('/pic/projects/im3/wm/Jim/pmp_input_files/crop_ids_by_farm.p', 'rb') as fp:
                    crop_ids_by_farm = pickle.load(fp)
                with open('/pic/projects/im3/wm/Jim/pmp_input_files/crop_ids_by_farm_and_constraint.p', 'rb') as fp:
                    crop_ids_by_farm_and_constraint = pickle.load(fp)
                with open('/pic/projects/im3/wm/Jim/pmp_input_files/max_land_constr_20220307_protocol2.p', 'rb') as fp:
                    land_constraints_by_farm = pickle.load(fp)
                # with open('/pic/projects/im3/wm/Jim/pmp_input_files/sw_gw_constr_20220405/sw_calib_constraints_202203319_protocol2.p', 'rb') as fp:
                #     sw_constraints_by_farm = pickle.load(fp)
                with open('/pic/projects/im3/wm/Jim/pmp_input_files/sw_gw_constr_20220405/gw_calib_constraints_20220401_protocol2.p', 'rb') as fp:
                    gw_constraints_by_farm = pickle.load(fp)

                # Revise to account for removal of "Fodder_Herb category"
                crop_ids_by_farm_new = {}
                for i in crop_ids_by_farm:
                    crop_ids_by_farm_new[i] = crop_ids_by_farm[i][0:10]
                crop_ids_by_farm = crop_ids_by_farm_new
                crop_ids_by_farm_and_constraint = crop_ids_by_farm_new

                # Load gammas, net prices, etc
                with open('/pic/projects/im3/wm/Jim/pmp_input_files/sw_gw_constr_20220408/gammas_total_dict_20220408_protocol2.p', 'rb') as fp:
                    gammas_total = pickle.load(fp)
                with open('/pic/projects/im3/wm/Jim/pmp_input_files/sw_gw_constr_20220408/net_prices_total_dict_20220408_protocol2.p', 'rb') as fp:
                    net_prices_total = pickle.load(fp)
                with open('/pic/projects/im3/wm/Jim/pmp_input_files/sw_gw_constr_20220408//alphas_total_dict_20220408_protocol2.p', 'rb') as fp:
                    alphas_total = pickle.load(fp)
                with open('/pic/projects/im3/wm/Jim/pmp_input_files/net_prices_sw_20220323_protocol2.p', 'rb') as fp:
                    net_prices_sw = pickle.load(fp)
                # with open('/pic/projects/im3/wm/Jim/pmp_input_files/alphas_sw_20220323_protocol2.p', 'rb') as fp:
                    # alphas_sw = pickle.load(fp)
                # with open('/pic/projects/im3/wm/Jim/pmp_input_files/gammas_sw_20220323_protocol2.p', 'rb') as fp:
                    # gammas_sw = pickle.load(fp)
                with open('/pic/projects/im3/wm/Jim/pmp_input_files/net_prices_gw_20220323_protocol2.p', 'rb') as fp:
                    net_prices_gw = pickle.load(fp)

                x_start_values=dict(enumerate([0.0]*3))

                logging.info('I have loaded constructed model indices,constraints for month, year: ' + month + ' ' + year)

                ## C.2. 2st stage: Quadratic model included in JWP model simulations
                ## C.2.a. Constructing model inputs:
                ##  (repetition to be safe - deepcopy does not work on PYOMO models)
                chunk_size = 555  # JY: for a total of 97 chunks (53,835 farms / 555)
                no_of_chunks = len(farm_ids) / chunk_size

                first = True
                for n in range(int(no_of_chunks)):

                    print('starting chunk: ' + str(n))
                    # subset farm ids
                    farm_ids_subset = list(range(chunk_size * n, chunk_size * (n + 1)))

                    # subset crop ids
                    crop_ids_by_farm_subset = {key: crop_ids_by_farm[key] for key in farm_ids_subset}
                    ids_subset = []
                    for key, list_value in crop_ids_by_farm_subset.items():
                        for value in list_value:
                            ids_subset.append(value)
                    ids_subset_sorted = sorted(ids_subset)

                    # subset various dictionaries;
                    keys_to_extract = list(range(chunk_size * n, chunk_size * (
                                n + 1)))  # will multiply start and end values by n+1 once integrated in loop
                    net_prices_total_subset = {key: net_prices_total[key] for key in ids_subset_sorted}
                    net_prices_sw_subset = {key: net_prices_sw[key] for key in ids_subset_sorted}
                    net_prices_gw_subset = {key: net_prices_gw[key] for key in ids_subset_sorted}
                    gammas_total_subset = {key: gammas_total[key] for key in ids_subset_sorted}
                    nirs_subset = {key: nirs[key] for key in ids_subset_sorted}
                    # alphas_sw_subset = {key: alphas_sw[key] for key in farm_ids_subset}
                    # gammas_sw_subset = {key: gammas_sw[key] for key in farm_ids_subset}
                    land_constraints_by_farm_subset = {key: land_constraints_by_farm[key] for key in farm_ids_subset}
                    # water_constraints_by_farm_subset = {key: water_constraints_by_farm[key] for key in farm_ids_subset}
                    gw_constraints_by_farm_subset = {key: gw_constraints_by_farm[key] for key in farm_ids_subset}
                    sw_constraints_by_farm_subset = {key: sw_constraints_by_farm[key] for key in farm_ids_subset}
                    # water_constraints_by_farm_subset[36335] = 63026538.58  ### JY TEMP

                    # set price to zero for gammas that are zero
                    for key,value in gammas_total_subset.items():
                        if value == 0:
                            net_prices_total_subset[key] = -9999999999

                    ## C.2. 2st stage: Quadratic model included in JWP model simulations
                    ## C.2.a. Constructing model inputs:
                    ##  (repetition to be safe - deepcopy does not work on PYOMO models)
                    ## C.1. Constructing model inputs:
                    fwm_s = ConcreteModel()
                    fwm_s.ids = Set(initialize=ids_subset_sorted)
                    fwm_s.farm_ids = Set(initialize=farm_ids_subset)
                    fwm_s.crop_ids_by_farm = Set(fwm_s.farm_ids, initialize=crop_ids_by_farm_subset)
                    fwm_s.crop_ids_by_farm_and_constraint = Set(fwm_s.farm_ids, initialize=crop_ids_by_farm_subset)
                    fwm_s.net_prices_sw = Param(fwm_s.ids, initialize=net_prices_sw_subset, mutable=True)
                    fwm_s.net_prices_total = Param(fwm_s.ids, initialize=net_prices_total_subset, mutable=True)
                    fwm_s.net_prices_gw = Param(fwm_s.ids, initialize=net_prices_gw_subset, mutable=True)
                    fwm_s.gammas_total = Param(fwm_s.ids, initialize=gammas_total_subset, mutable=True)
                    # fwm_s.alphas_sw = Param(fwm_s.farm_ids, initialize=alphas_sw_subset, mutable=True)
                    # fwm_s.gammas_sw = Param(fwm_s.farm_ids, initialize=gammas_sw_subset, mutable=True)
                    fwm_s.land_constraints = Param(fwm_s.farm_ids, initialize=land_constraints_by_farm_subset, mutable=True)
                    # fwm_s.water_constraints = Param(fwm_s.farm_ids, initialize=water_constraints_by_farm_subset,
                                                    # mutable=True)
                    fwm_s.sw_constraints = Param(fwm_s.farm_ids, initialize=sw_constraints_by_farm_subset,
                                                    mutable=True)
                    fwm_s.gw_constraints = Param(fwm_s.farm_ids, initialize=gw_constraints_by_farm_subset,
                                                    mutable=True)
                    fwm_s.xs_total = Var(fwm_s.ids, domain=NonNegativeReals, initialize=x_start_values)
                    fwm_s.xs_sw = Var(fwm_s.ids, domain=NonNegativeReals, initialize=x_start_values)
                    fwm_s.xs_gw = Var(fwm_s.ids, domain=NonNegativeReals, initialize=x_start_values)
                    # obs_lu_total = dict(data_profit["area_irrigated"])
                    # obs_lu_sw = dict(area_irrigated_sw_farm["area_irrigated_sw"])
                    # fwm_s.obs_lu_total = Param(fwm_s.ids, initialize=obs_lu_total, mutable=True)
                    # fwm_s.obs_lu_sw = Param(fwm_s.ids, initialize=obs_lu_sw, mutable=True)
                    fwm_s.nirs = Param(fwm_s.ids, initialize=nirs_subset, mutable=True)

                    ## C.2. Constructing model functions:
                    def obj_fun(fwm_s):
                        return 0.00001 * sum(sum((fwm_s.net_prices_total[h] * fwm_s.xs_total[h] - 0.5 * fwm_s.gammas_total[h] * fwm_s.xs_total[h] * fwm_s.xs_total[h]) for h in fwm_s.crop_ids_by_farm[f]) +
                            sum((fwm_s.net_prices_sw[i] * fwm_s.xs_sw[i]) for i in fwm_s.crop_ids_by_farm[f]) +
                            sum((fwm_s.net_prices_gw[g] * fwm_s.xs_gw[g]) for g in fwm_s.crop_ids_by_farm[f]) for f in fwm_s.farm_ids)  # JY double check this!

                        # return 0.00001 * sum(sum((fwm_s.net_prices_total[h] * fwm_s.xs_total[h] - 0.5 * fwm_s.gammas_total[
                        #     h] * fwm_s.xs_total[h] * fwm_s.xs_total[h]) for h in fwm_s.crop_ids_by_farm[f]) +
                        #                      sum((fwm_s.net_prices_sw[i] * fwm_s.xs_sw[i]) for i in
                        #                          fwm_s.crop_ids_by_farm[f]) +
                        #                      sum((fwm_s.net_prices_gw[g] * fwm_s.xs_gw[g]) for g in
                        #                          fwm_s.crop_ids_by_farm[f]) -
                        #                      (fwm_s.alphas_sw[f] * sum(fwm_s.xs_sw[s] for s in fwm_s.crop_ids_by_farm[f])) -
                        #                      (0.5 * fwm_s.gammas_sw[f] * sum(
                        #                          fwm_s.xs_sw[t] for t in fwm_s.crop_ids_by_farm[f])) * sum(
                        #     fwm_s.xs_sw[u] for u in fwm_s.crop_ids_by_farm[f]) for f in
                        #                      fwm_s.farm_ids)  # JY double check this!

                    fwm_s.obj_f = Objective(rule=obj_fun, sense=maximize)

                    def land_constraint(fwm_s, ff):
                        return sum(fwm_s.xs_total[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= \
                               fwm_s.land_constraints[ff]

                    fwm_s.c1 = Constraint(fwm_s.farm_ids, rule=land_constraint)

                    def obs_lu_constraint_sum(fwm_s, i):
                        return fwm_s.xs_sw[i] + fwm_s.xs_gw[i] == fwm_s.xs_total[i]

                    fwm_s.c5 = Constraint(fwm_s.ids, rule=obs_lu_constraint_sum)

                    # def water_constraint(fwm_s, ff):
                    #     return sum(fwm_s.xs_sw[i]*fwm_s.nirs[i]*(1000/1000) for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.water_constraints[ff]  # JY multiplication by 1,000 gets us back to non-scaled NIR values, and divide 1,000 to account for scaling of areas
                    # fwm_s.c2 = Constraint(fwm_s.farm_ids, rule=water_constraint)

                    def sw_constraint(fwm_s, ff):
                        return sum(fwm_s.xs_sw[i]*fwm_s.nirs[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.sw_constraints[ff]
                    fwm_s.c2 = Constraint(fwm_s.farm_ids, rule=sw_constraint)

                    def gw_constraint(fwm_s, ff):
                        return sum(fwm_s.xs_gw[i]*fwm_s.nirs[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.gw_constraints[ff]
                    fwm_s.c6 = Constraint(fwm_s.farm_ids, rule=gw_constraint)

                    logging.info('I have successfully constructed pyomo model for month, year, chunk: ' + month + ' ' + year + ' ' + str(n))

                    ## C.2.c Creating and running the solver:
                    # start_time = datetime.datetime.now()
                    try:
                        opt = SolverFactory("ipopt", solver_io='nl')
                        results = opt.solve(fwm_s, keepfiles=False, tee=True)
                        print(results.solver.termination_condition)
                    except:
                        logging.info('Pyomo model solve has failed for month, year: ' + month + ' ' + year)

                    ## D.1. Storing main model outputs:
                    result_xs_sw = dict(fwm_s.xs_sw.get_values())
                    result_xs_gw = dict(fwm_s.xs_gw.get_values())
                    result_xs_total = dict(fwm_s.xs_total.get_values())
                    logging.info('Extracted results from Pyomo')

                    # convert result_xs_sw to pandas dataframe and join to data_profit
                    if first is True:
                        results_pd = data_profit
                        results_pd['xs_gw'] = 0
                        results_pd['xs_sw'] = 0
                        results_pd['xs_total'] = 0
                        results_pd['id'] = results_pd['index']
                        first = False
                    results_xs_sw_pd = pd.DataFrame.from_dict(result_xs_sw, orient='index')
                    results_xs_sw_pd['id'] = results_xs_sw_pd.index + 1
                    results_xs_sw_pd = results_xs_sw_pd.rename(columns={0: "xs_sw_temp"})
                    results_pd = results_pd.merge(results_xs_sw_pd[['id', 'xs_sw_temp']], how='left', on=['id'])
                    results_pd.loc[results_pd['xs_sw_temp'].notnull(), 'xs_sw'] = results_pd['xs_sw_temp']
                    results_xs_gw_pd = pd.DataFrame.from_dict(result_xs_gw, orient='index')
                    results_xs_gw_pd['id'] = results_xs_gw_pd.index + 1
                    results_xs_gw_pd = results_xs_gw_pd.rename(columns={0: "xs_gw_temp"})
                    results_pd = results_pd.merge(results_xs_gw_pd[['id', 'xs_gw_temp']], how='left', on=['id'])
                    results_pd.loc[results_pd['xs_gw_temp'].notnull(), 'xs_gw'] = results_pd['xs_gw_temp']
                    results_xs_total_pd = pd.DataFrame.from_dict(result_xs_total, orient='index')
                    results_xs_total_pd['id'] = results_xs_total_pd.index + 1
                    results_xs_total_pd = results_xs_total_pd.rename(columns={0: "xs_total_temp"})
                    results_pd = results_pd.merge(results_xs_total_pd[['id', 'xs_total_temp']], how='left', on=['id'])
                    results_pd.loc[results_pd['xs_total_temp'].notnull(), 'xs_total'] = results_pd['xs_total_temp']
                    results_pd = results_pd.drop(['xs_gw_temp', 'xs_sw_temp', 'xs_total_temp'], axis=1)
                    logging.info('Added results to pandas dataframe')

                # JY store results into a pandas dataframe
                # JY store results into a pandas dataframe
                # results_pd = results_pd.assign(calc_area=result_xs.values())
                results_pd['calc_gw_demand'] = results_pd['xs_gw'] * results_pd['nir_corrected'] / 25583.64  # unit conversion from acre-ft/year to m3/s; calc area [acres], nir [acre-ft/acres/year]
                results_pd['calc_sw_demand'] = results_pd['xs_sw'] * results_pd['nir_corrected'] / 25583.64  # unit conversion from acre-ft/year to m3/s; calc area [acres], nir [acre-ft/acres/year]
                results_pd['calc_total_demand'] = results_pd['xs_total'] * results_pd['nir_corrected'] / 25583.64  # unit conversion from acre-ft/year to m3/s; calc area [acres], nir [acre-ft/acres/year]
                results_pivot = pd.pivot_table(results_pd, index=['nldas'], values=['calc_gw_demand', 'calc_sw_demand', 'calc_total_demand'], aggfunc=np.sum)  #JY demand is order of magnitude low, double check calcs
                logging.info('Compiled results from all chunks and calculated demand')

                results_pd = results_pd[['nldas', 'crop', 'xs_gw', 'xs_sw', 'xs_total', 'nir_corrected']]
                results_pd.to_csv('/pic/scratch/yoon644/csmruns/wm_abm_run/run/abm_results_' + str(year_int))

                # read a sample water demand input file
                file = '/pic/projects/im3/wm/Jim/pmp_input_files/RCP8.5_GCAM_water_demand_1980_01_copy.nc'
                with netCDF4.Dataset(file, 'r') as nc:
                    # for key, var in nc.variables.items():
                    #     print(key, var.dimensions, var.shape, var.units, var.long_name, var._FillValue)

                    lat = nc['lat'][:]
                    lon = nc['lon'][:]
                    demand = nc['totalDemand'][:]

                # read NLDAS grid reference file
                df_grid = pd.read_csv('/pic/projects/im3/wm/Jim/pmp_input_files/NLDAS_Grid_Reference.csv')

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
                try:
                    results_pivot = results_pivot.set_index('nldas',drop=False)
                except KeyError:
                    pass

                # read ABM values into df_nc basing on the same index
                df_nc.loc[results_pivot.index,'totalDemand'] = results_pivot.calc_sw_demand.values

                if year_int<=1940:
                    for yr in range(70):
                        for month in months:
                            year_out = year_int + yr + 1
                            str_year = str(year_out)
                            new_fname = '/pic/projects/im3/wm/Jim/pmp_input_files/demand_input/RCP8.5_GCAM_water_demand_'+ str_year + '_' + month + '.nc' # define ABM demand input directory
                            shutil.copyfile(file, new_fname)
                            demand_ABM = df_nc.totalDemand.values.reshape(len(lat),len(lon),order='C')
                            with netCDF4.Dataset(new_fname,'a') as nc:
                                nc['totalDemand'][:] = np.ma.masked_array(demand_ABM,mask=nc['totalDemand'][:].mask)

                else:
                    for month in months:
                        str_year = str(year_int)
                        new_fname = '/pic/projects/im3/wm/Jim/pmp_input_files/demand_input/RCP8.5_GCAM_water_demand_'+ str_year + '_' + month + '.nc' # define ABM demand input directory
                        shutil.copyfile(file, new_fname)
                        demand_ABM = df_nc.totalDemand.values.reshape(len(lat),len(lon),order='C')
                        with netCDF4.Dataset(new_fname,'a') as nc:
                            nc['totalDemand'][:] = np.ma.masked_array(demand_ABM,mask=nc['totalDemand'][:].mask)

                logging.info('I have successfully written out new demand files for month, year: ' + month + ' ' + year)

            else:
                pass
    except Exception as e:
        logging.exception(str(e))