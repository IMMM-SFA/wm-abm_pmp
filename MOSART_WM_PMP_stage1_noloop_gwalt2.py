# pmp calibrated costs for both sw and gw

import os
import sys
global basepath
print(os.path.dirname(sys.argv[0]))
##basepath = os.path.dirname(sys.argv[0]).split(__file__)[0]
from pyomo.environ import * # JY temp
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
import pdb


#data_file=pd.ExcelFile("data_inputs/MOSART_WM_PMP_inputs_v1.xlsx")
#data_file=pd.ExcelFile("data_inputs/MOSART_WM_PMP_inputs_20201005.xlsx")
#data_file=pd.ExcelFile("data_inputs/MOSART_WM_PMP_inputs_20201028_GW.xlsx")
#data_file=pd.ExcelFile("data_inputs/MOSART_WM_PMP_inputs_20220223_GW.xlsx")
#data_file=pd.ExcelFile("data_inputs/MOSART_WM_PMP_inputs_20220311_GW.xlsx")
data_file=pd.ExcelFile("data_inputs/MOSART_WM_PMP_inputs_20220323_GW.xlsx")
data_profit = data_file.parse("Profit")
data_profit['area_irrigated'] = data_profit['area_irrigated'] * 1000
data_profit['area_irrigated_gw'] = data_profit['area_irrigated_gw'] * 1000
data_profit['area_irrigated_sw'] = data_profit['area_irrigated_sw'] * 1000
aggregation_functions = {'area_irrigated_sw': 'sum'}
area_irrigated_sw_farm = data_profit.groupby(['nldas'], as_index=False).aggregate(aggregation_functions)
aggregation_functions = {'area_irrigated_gw': 'sum'}
area_irrigated_gw_farm = data_profit.groupby(['nldas'], as_index=False).aggregate(aggregation_functions)

#data_constraint = data_file.parse("Constraint")

nldas_ids=data_profit["nldas"][0:53835].tolist()

## B.1. Preparing model indices and constraints:
#ids = range(592185) # total number of crop and nldas ID combinations
ids = range(538350) # total number of crop and nldas ID combinations
farm_ids = range(53835) # total number of farm agents / nldas IDs
sd_no = len(farm_ids)
crop_types=[str(i) for i in list(pd.unique(data_profit["crop"]))]
crop_no=len(crop_types)
crop_ids_by_farm_and_constraint={}
land_constraints_by_farm={}
water_constraints_by_farm={}
#crop_ids_by_farm=dict(enumerate([np.where(data_profit["nldas"]==nldas_ids[i])[0].tolist() for i in range(53835)])) #JY this takes forever, find better way
with open('data_inputs/pickles/crop_ids_by_farm.p', 'rb') as fp:
    crop_ids_by_farm = pickle.load(fp)
# with open('data_inputs/max_land_constr_20201102.p', 'rb') as fp:
#     land_constraints_by_farm = pickle.load(fp, encoding='latin1')
with open('data_inputs/pickles/max_land_constr_20220307_protocol2.p', 'rb') as fp:
    land_constraints_by_farm = pickle.load(fp, encoding='latin1')
with open('data_inputs/pickles/water_constraints_by_farm_v2.p', 'rb') as fp:
    water_constraints_by_farm = pickle.load(fp, encoding='latin1')
with open('data_inputs/pickles/crop_ids_by_farm_and_constraint.p', 'rb') as fp:
    crop_ids_by_farm_and_constraint = pickle.load(fp)

#Revise to account for removal of "Fodder_Herb category"
crop_ids_by_farm_new = {}
for i in crop_ids_by_farm:
    crop_ids_by_farm_new[i] = crop_ids_by_farm[i][0:10]
crop_ids_by_farm = crop_ids_by_farm_new
crop_ids_by_farm_and_constraint = crop_ids_by_farm_new

water_constraints_by_farm = dict.fromkeys(water_constraints_by_farm, 9999999999)

# JY this outputs pandas series for each entry of land_constraints_by_farm and water_constraints_by_farm; need
# to convert to float
# for i in range(53835):
#     #crop_ids_by_farm_and_constraint[i]=np.where((data_profit["nldas"]==nldas_ids[i]))[0].tolist()
#     constraint_ids=np.where((data_constraint["nldas"]==nldas_ids[i]))[0]
#     #land_constraints_by_farm[i]=data_constraint.iloc[constraint_ids]["land_constraint"].astype('float')
#     water_constraints_by_farm_2[i]=data_constraint.iloc[constraint_ids]["water_constraint"].astype('float')

# Replace NIRs with 0 value to 0.1 (nominal irrigation) to account for inconsistency between observed surface water irrigated area and NIR

## B.2. Preparing linear profit coefficients:
prices=data_profit["price"]
yields=data_profit["yield"]
land_costs=data_profit["land_cost"]
water_nirs=data_profit["nir_corrected"]
sw_costs=data_profit["sw_cost"] #JY need to fill in gaps for DC, figure out how to handle groundwater costs
gw_costs=data_profit["gw_cost"]
data_profit["alpha"] = 0
alphas_total=data_profit["alpha"]
alphas_sw=data_profit["alpha"].head(53835)  # ids for surface water are farm specific (summed over all crops)

# linear_term_sum=[p*y - c - (swc*n) - (gwc*n) - aland - asw for p,y,c,swc,gwc,n,aland,asw in zip(prices,yields,land_costs,sw_costs,gw_costs,water_nirs,alphas_land,alphas_sw)]
linear_term_sum_total = [p*y - c - aland for p,y,c,aland in zip(prices,yields,land_costs,alphas_total)]
linear_term_sum_sw = [-(swc*n*1000) for swc,n in zip(sw_costs,water_nirs)]  # multiply by 1000 to get to non-corrected NIRs
linear_term_sum_gw = [-(gwc*n*1000) for gwc,n in zip(gw_costs,water_nirs)]
# linear_term_sum_gw = [p*y - c - (gwc*n) - aland for p,y,c,gwc,n,aland in zip(prices,yields,land_costs,gw_costs,water_nirs,alphas_total)]

## B.3. Preparing model vars and params: (these need to be dict()!)
gammas_total=dict(data_profit["gamma"]) #JY temporarily set at 100
gammas_sw=dict(data_profit["gamma"].head(53835)) #JY temporarily set at 100
net_prices_total=dict(enumerate(linear_term_sum_total))
net_prices_sw=dict(enumerate(linear_term_sum_sw))
net_prices_gw=dict(enumerate(linear_term_sum_gw))
x_start_values=dict(enumerate([0.0]*3))
nirs=dict(water_nirs)


## C.1. Constructing model inputs:
fwm = ConcreteModel()
fwm.ids = Set(initialize=ids)
# fwm.ids = Set(initialize=crop_ids_by_farm)
fwm.farm_ids = Set(initialize=farm_ids)
fwm.crop_ids_by_farm = Set(fwm.farm_ids, initialize=crop_ids_by_farm)
fwm.crop_ids_by_farm_and_constraint = Set(fwm.farm_ids, initialize=crop_ids_by_farm_and_constraint)
fwm.net_prices_total = Param(fwm.ids, initialize=net_prices_total, mutable=True)
fwm.net_prices_sw = Param(fwm.ids, initialize=net_prices_sw, mutable=True)
fwm.net_prices_gw = Param(fwm.ids, initialize=net_prices_gw, mutable=True)
fwm.gammas_total = Param(fwm.ids, initialize=gammas_total, mutable=True)
fwm.alphas_sw = Param(fwm.farm_ids, initialize=alphas_sw, mutable=True)
fwm.gammas_sw = Param(fwm.farm_ids, initialize=gammas_sw, mutable=True)
fwm.land_constraints = Param(fwm.farm_ids, initialize=land_constraints_by_farm, mutable=True)
fwm.water_constraints = Param(fwm.farm_ids, initialize=water_constraints_by_farm, mutable=True)
fwm.xs_total = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)
fwm.xs_sw = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)
fwm.xs_gw = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)
obs_lu_total = dict(data_profit["area_irrigated"])
obs_lu_sw = dict(area_irrigated_sw_farm["area_irrigated_sw"])
obs_lu_gw = dict(area_irrigated_gw_farm["area_irrigated_gw"])
crop_counter = 0
# for key,value in obs_lu_total.items():  # JY TEMP comment out
#     if crop_counter == 53835:
#         crop_counter = 0
#     if value == 0:
#         obs_lu_total[key] = .0001  # NEED TO ACCOUNT FOR THIS IN OBS_LU_SW below
#         obs_lu_sw[crop_counter] += .00005  # JY revised from .0001
#     crop_counter += 1
fwm.obs_lu_total = Param(fwm.ids, initialize=obs_lu_total, mutable=True)
fwm.obs_lu_sw = Param(fwm.farm_ids, initialize=obs_lu_sw, mutable=True)  # JY EDIT
fwm.obs_lu_gw = Param(fwm.farm_ids, initialize=obs_lu_gw, mutable=True)  # JY EDIT
fwm.nirs = Param(fwm.ids, initialize=nirs, mutable=True)

## C.2. Constructing model functions:  #!JY! test multiply by 1000 to get non-adjusted NIR
def obj_fun(fwm):
    return sum(sum((fwm.net_prices_total[h] * fwm.xs_total[h]) for h in fwm.crop_ids_by_farm[f]) +
               sum((fwm.net_prices_sw[i] * fwm.xs_sw[i]) for i in fwm.crop_ids_by_farm[f]) +
               sum((fwm.net_prices_gw[g] * fwm.xs_gw[g]) for g in fwm.crop_ids_by_farm[f]) for f in fwm.farm_ids)  # JY double check this!
fwm.obj_f = Objective(rule=obj_fun, sense=maximize)

# JY need to re-implement this
# def land_constraint(fwm, ff):
#     return sum(fwm.xs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff]) <= fwm.land_constraints[ff]
# fwm.c1 = Constraint(fwm.farm_ids, rule=land_constraint)


def obs_lu_constraint_total(fwm, i):
    return fwm.xs_total[i] == fwm.obs_lu_total[i]
fwm.c3 = Constraint(fwm.ids, rule=obs_lu_constraint_total)

def obs_lu_constraint_sw(fwm, f):
    return sum(fwm.xs_sw[i] for i in fwm.crop_ids_by_farm[f]) == fwm.obs_lu_sw[f]
fwm.c4 = Constraint(fwm.farm_ids, rule=obs_lu_constraint_sw)

def obs_lu_constraint_gw(fwm, f):  # JY ADD
    return sum(fwm.xs_gw[i] for i in fwm.crop_ids_by_farm[f]) == fwm.obs_lu_gw[f]
fwm.c6 = Constraint(fwm.farm_ids, rule=obs_lu_constraint_gw)

# def obs_lu_constraint_sw(fwm, i):
#     return fwm.xs_sw[i] == fwm.obs_lu_sw[i]
# fwm.c4 = Constraint(fwm.ids, rule=obs_lu_constraint_sw)

# def water_constraint(fwm, ff):
#     return sum(fwm.xs_sw[i]*fwm.nirs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff]) <= fwm.water_constraints[ff]
# fwm.c2 = Constraint(fwm.farm_ids, rule=water_constraint)

# def obs_lu_constraint_sum(fwm, i):
#     return fwm.xs_sw[i] + fwm.xs_gw[i] == fwm.xs_total[i]
# fwm.c5 = Constraint(fwm.ids, rule=obs_lu_constraint_sum)

fwm.dual = Suffix(direction=Suffix.IMPORT)
#
# def water_constraint(fwm, ff):
#     return sum(fwm.xs_sw[i]*fwm.nirs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff]) <= fwm.water_constraints[ff]
# fwm.c2 = Constraint(fwm.farm_ids, rule=water_constraint)

## C.3. Solve
opt = SolverFactory("ipopt", solver_io='nl')
results = opt.solve(fwm, keepfiles=False, tee=True)
print(results.solver.termination_condition)

## C.1.d. Save duals:
from pyomo.core import Constraint

obs_lu_duals_total = dict()
for c in fwm.component_objects(Constraint, active=True):
    if str(c) == "c3":
        cobject = getattr(fwm, str(c))
        for index in cobject:
            obs_lu_duals_total[index] = fwm.dual[cobject[index]]

obs_lu_duals_sw = dict()
for c in fwm.component_objects(Constraint, active=True):
    if str(c) == "c4":
        cobject = getattr(fwm, str(c))
        for index in cobject:
            obs_lu_duals_sw[index] = fwm.dual[cobject[index]]

obs_lu_duals_gw = dict()
for c in fwm.component_objects(Constraint, active=True):
    if str(c) == "c6":
        cobject = getattr(fwm, str(c))
        for index in cobject:
            obs_lu_duals_gw[index] = fwm.dual[cobject[index]]

# ## C.1.e. 1st stage result: Calculate alpha and gamma:
# # gamma1 = [((2. * a / b) if (b > 0.0) else 0.0) for a,b in zip(obs_lu_duals.values(),obs_lu.values())]
# gamma1 = [((a / b) if (b > 0.0) else 0.0) for a, b in zip(obs_lu_duals.values(), obs_lu.values())]
# alpha1 = [-(0.5 * a * b) for a, b in zip(gamma1, obs_lu.values())]
# print(alpha1)
# print("+++ alpha check: +++")
# print([(a, b) for a, b in zip(alphas, alpha1)])
# print("+++ gamma check: +++")
# print([(a, b) for a, b in zip(gammas, gamma1)])
# alphas = alpha1
# linear_term_sum=[p*y - c - wc*n - a for p,y,c,wc,n,a in zip(prices,yields,land_costs,sw_costs,water_nirs,alphas)]
# gammas = dict(enumerate(gamma1))
# net_prices = dict(enumerate(linear_term_sum))

## C.1.e. 1st stage result: Calculate alpha and gamma:
# gamma1 = [((2. * a / b) if (b > 0.0) else 0.0) for a,b in zip(obs_lu_duals.values(),obs_lu.values())]
gamma1_total = [((2. * a / b) if (b > 0.0) else 0.0) for a, b in zip(obs_lu_duals_total.values(), obs_lu_total.values())]
alpha1_total = [-(0.5 * a * b) for a, b in zip(gamma1_total, obs_lu_total.values())]
print(alpha1_total)
print("+++ alpha check: +++")
print([(a, b) for a, b in zip(alphas_total, alpha1_total)])
print("+++ gamma check: +++")
print([(a, b) for a, b in zip(gammas_total, gamma1_total)])
alphas_total = alpha1_total

linear_term_sum_total = [p*y - c - aland for p,y,c,aland in zip(prices, yields, land_costs, alphas_total)]
gammas_total = dict(enumerate(gamma1_total))

net_prices_total = dict(enumerate(linear_term_sum_total))

gamma1_sw = [((2. * a / b) if (b > 0.0) else 0.0) for a, b in zip(obs_lu_duals_sw.values(), obs_lu_sw.values())]
alpha1_sw = [-(0.5 * a * b) for a, b in zip(gamma1_sw, obs_lu_sw.values())]
print(alpha1_sw)
print("+++ alpha check: +++")
print([(a, b) for a, b in zip(alphas_sw, alpha1_sw)])
print("+++ gamma check: +++")
print([(a, b) for a, b in zip(gammas_sw, gamma1_sw)])
alphas_sw = dict(enumerate(alpha1_sw))
gammas_sw = dict(enumerate(gamma1_sw))

gamma1_gw = [((2. * a / b) if (b > 0.0) else 0.0) for a, b in zip(obs_lu_duals_gw.values(), obs_lu_gw.values())]
alpha1_gw = [-(0.5 * a * b) for a, b in zip(gamma1_gw, obs_lu_gw.values())]
print(alpha1_gw)
alphas_gw = dict(enumerate(alpha1_gw))
gammas_gw = dict(enumerate(gamma1_gw))

import datetime
start_time = datetime.datetime.now()
# chunk_size = 555  # JY temp, eventually convert into a loop
chunk_size = 1
no_of_chunks = len(farm_ids) / chunk_size

first = True
for n in [1]:
# for n in range(int(1)):
# for n in range(int(no_of_chunks)):
    print('starting chunk: ' + str(n))
    # subset farm ids
    # farm_ids_subset = list(range(chunk_size*n, chunk_size*(n+1)))
    farm_ids_subset = list(range(36335, 36336))
    # subset crop ids
    crop_ids_by_farm_subset = {key: crop_ids_by_farm[key] for key in farm_ids_subset}
    ids_subset = []
    for key,list_value in crop_ids_by_farm_subset.items():
        for value in list_value:
            ids_subset.append(value)
    ids_subset_sorted = sorted(ids_subset)

    # subset various dictionaries;
    keys_to_extract = list(range(chunk_size*n, chunk_size*(n+1)))  # will multiply start and end values by n+1 once integrated in loop
    net_prices_total_subset = {key: net_prices_total[key] for key in ids_subset_sorted}
    net_prices_sw_subset = {key: net_prices_sw[key] for key in ids_subset_sorted}
    net_prices_gw_subset = {key: net_prices_gw[key] for key in ids_subset_sorted}
    # net_prices_gw_subset.update((x, y*2) for x, y in net_prices_gw_subset.items())  ### JY TEMP
    gammas_total_subset = {key: gammas_total[key] for key in ids_subset_sorted}
    nirs_subset = {key: nirs[key] for key in ids_subset_sorted}
    alphas_sw_subset = {key: alphas_sw[key] for key in farm_ids_subset}
    gammas_sw_subset = {key: gammas_sw[key] for key in farm_ids_subset}
    alphas_gw_subset = {key: alphas_gw[key] for key in farm_ids_subset}
    gammas_gw_subset = {key: gammas_gw[key] for key in farm_ids_subset}
    land_constraints_by_farm_subset = {key: land_constraints_by_farm[key] for key in farm_ids_subset}
    water_constraints_by_farm_subset = {key: water_constraints_by_farm[key] for key in farm_ids_subset}
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
    fwm_s.alphas_sw = Param(fwm_s.farm_ids, initialize=alphas_sw_subset, mutable=True)
    fwm_s.gammas_sw = Param(fwm_s.farm_ids, initialize=gammas_sw_subset, mutable=True)
    fwm_s.alphas_gw = Param(fwm_s.farm_ids, initialize=alphas_gw_subset, mutable=True)
    fwm_s.gammas_gw = Param(fwm_s.farm_ids, initialize=gammas_gw_subset, mutable=True)
    fwm_s.land_constraints = Param(fwm_s.farm_ids, initialize=land_constraints_by_farm_subset, mutable=True)
    fwm_s.water_constraints = Param(fwm_s.farm_ids, initialize=water_constraints_by_farm_subset, mutable=True)
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
                   sum((fwm_s.net_prices_gw[g] * fwm_s.xs_gw[g]) for g in fwm_s.crop_ids_by_farm[f]) -
                   (fwm_s.alphas_sw[f] * sum(fwm_s.xs_sw[s] for s in fwm_s.crop_ids_by_farm[f])) -
                   (fwm_s.alphas_gw[f] * sum(fwm_s.xs_gw[s] for s in fwm_s.crop_ids_by_farm[f])) -
                   (0.5 * fwm_s.gammas_sw[f] * sum(fwm_s.xs_sw[t] for t in fwm_s.crop_ids_by_farm[f])) * sum(fwm_s.xs_sw[u] for u in fwm_s.crop_ids_by_farm[f]) -
                   (0.5 * fwm_s.gammas_gw[f] * sum(fwm_s.xs_gw[t] for t in fwm_s.crop_ids_by_farm[f])) * sum(fwm_s.xs_gw[u] for u in fwm_s.crop_ids_by_farm[f]) for f in fwm_s.farm_ids)  # JY double check this!
    fwm_s.obj_f = Objective(rule=obj_fun, sense=maximize)

    # def land_constraint(fwm_s, ff):
    #     return sum(fwm_s.xs_total[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.land_constraints[ff]
    # fwm_s.c1 = Constraint(fwm_s.farm_ids, rule=land_constraint)

    def obs_lu_constraint_sum(fwm_s, i):
        return fwm_s.xs_sw[i] + fwm_s.xs_gw[i] == fwm_s.xs_total[i]
    fwm_s.c5 = Constraint(fwm_s.ids, rule=obs_lu_constraint_sum)

    def water_constraint(fwm_s, ff):
        return sum(fwm_s.xs_sw[i]*fwm_s.nirs[i]*1000 for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.water_constraints[ff]
    fwm_s.c2 = Constraint(fwm_s.farm_ids, rule=water_constraint)

    ## C.2.c Creating and running the solver:
    # start_time = datetime.datetime.now()
    opt = SolverFactory("ipopt", solver_io='nl')
    results = opt.solve(fwm_s, keepfiles=False, tee=True)
    print(results.solver.termination_condition)
    end_time = datetime.datetime.now()

    ## D.1. Storing main model outputs:
    result_xs_sw = dict(fwm_s.xs_sw.get_values())
    result_xs_gw = dict(fwm_s.xs_gw.get_values())
    result_xs_total = dict(fwm_s.xs_total.get_values())

    # JY results stored as pickle file (results_xs.p). Start here and load pickle files.
    with open('result_xs.p', 'rb') as fp:
        result_xs = pickle.load(fp)

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
    results_pd = results_pd.merge(results_xs_sw_pd[['id','xs_sw_temp']], how='left', on=['id'])
    results_pd.loc[results_pd['xs_sw_temp'].notnull(), 'xs_sw'] = results_pd['xs_sw_temp']
    results_xs_gw_pd = pd.DataFrame.from_dict(result_xs_gw, orient='index')
    results_xs_gw_pd['id'] = results_xs_gw_pd.index + 1
    results_xs_gw_pd = results_xs_gw_pd.rename(columns={0: "xs_gw_temp"})
    results_pd = results_pd.merge(results_xs_gw_pd[['id','xs_gw_temp']], how='left', on=['id'])
    results_pd.loc[results_pd['xs_gw_temp'].notnull(), 'xs_gw'] = results_pd['xs_gw_temp']
    results_xs_total_pd = pd.DataFrame.from_dict(result_xs_total, orient='index')
    results_xs_total_pd['id'] = results_xs_total_pd.index + 1
    results_xs_total_pd = results_xs_total_pd.rename(columns={0: "xs_total_temp"})
    results_pd = results_pd.merge(results_xs_total_pd[['id','xs_total_temp']], how='left', on=['id'])
    results_pd.loc[results_pd['xs_total_temp'].notnull(), 'xs_total'] = results_pd['xs_total_temp']
    results_pd = results_pd.drop(['xs_gw_temp', 'xs_sw_temp', 'xs_total_temp'], axis=1)

# JY store results into a pandas dataframe
results_pd = results_pd.assign(calc_area=result_xs.values())
results_pd['calc_gw_demand'] = results_pd['xs_gw'] * results_pd['nir_corrected'] / 25583.64  # unit conversion from acre-ft/year to m3/s; calc area [acres], nir [acre-ft/acres/year]
results_pd['calc_sw_demand'] = results_pd['xs_sw'] * results_pd['nir_corrected'] / 25583.64  # unit conversion from acre-ft/year to m3/s; calc area [acres], nir [acre-ft/acres/year]
results_pd['calc_total_demand'] = results_pd['xs_total'] * results_pd['nir_corrected'] / 25583.64  # unit conversion from acre-ft/year to m3/s; calc area [acres], nir [acre-ft/acres/year]
results_pivot = pd.pivot_table(results_pd, index=['nldas'], values=['calc_gw_demand', 'calc_sw_demand', 'calc_total_demand'], aggfunc=np.sum)  #JY demand is order of magnitude low, double check calcs

# JY export results to csv
results_pd = results_pd[['nldas','crop','xs_gw','xs_sw','xs_total','nir_corrected']]
results_pd.to_csv('/pic/scratch/yoon644/csmruns/wm_abm_run/run/abm_results_'+ str(year_int))

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

for month in months:
    str_year = str(year_int)
    new_fname = '/pic/projects/im3/wm/Jim/pmp_input_files/demand_input/RCP8.5_GCAM_water_demand_'+ str_year + '_' + month + '.nc' # define ABM demand input directory
    shutil.copyfile(file, new_fname)
    demand_ABM = df_nc.totalDemand.values.reshape(len(lat),len(lon),order='C')
    with netCDF4.Dataset(new_fname,'a') as nc:
        nc['totalDemand'][:] = np.ma.masked_array(demand_ABM,mask=nc['totalDemand'][:].mask)

logging.info('I have successfully written out new demand files for month, year: ' + month + ' ' + year)