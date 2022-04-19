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
data_file=pd.ExcelFile("data_inputs/MOSART_WM_PMP_inputs_20220323_GW.xlsx")
data_profit = data_file.parse("Profit")
data_profit['area_irrigated'] = data_profit['area_irrigated'] * 1000
data_profit['area_irrigated_gw'] = data_profit['area_irrigated_gw'] * 1000
data_profit['area_irrigated_sw'] = data_profit['area_irrigated_sw'] * 1000
aggregation_functions = {'area_irrigated_sw': 'sum'}
area_irrigated_sw_farm = data_profit.groupby(['nldas'], as_index=False).aggregate(aggregation_functions)

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
# with open('data_inputs/pickles/water_constraints_by_farm_v2.p', 'rb') as fp:
#     water_constraints_by_farm = pickle.load(fp, encoding='latin1')
with open('data_inputs/pickles/sw_gw_constr_20220321/sw_calib_constraints_202203319_protocol2.p', 'rb') as fp:
    sw_constraints_by_farm = pickle.load(fp, encoding='latin1')
with open('data_inputs/pickles/sw_gw_constr_20220321/gw_calib_constraints_202203319_protocol2.p', 'rb') as fp:
    gw_constraints_by_farm = pickle.load(fp, encoding='latin1')

with open('data_inputs/pickles/crop_ids_by_farm_and_constraint.p', 'rb') as fp:
    crop_ids_by_farm_and_constraint = pickle.load(fp)

#Revise to account for removal of "Fodder_Herb category"
crop_ids_by_farm_new = {}
for i in crop_ids_by_farm:
    crop_ids_by_farm_new[i] = crop_ids_by_farm[i][0:10]
crop_ids_by_farm = crop_ids_by_farm_new
crop_ids_by_farm_and_constraint = crop_ids_by_farm_new

# water_constraints_by_farm = dict.fromkeys(water_constraints_by_farm, 9999999999)

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
# alphas_sw=data_profit["alpha"].head(53835)  # ids for surface water are farm specific (summed over all crops)

# linear_term_sum=[p*y - c - (swc*n) - (gwc*n) - aland - asw for p,y,c,swc,gwc,n,aland,asw in zip(prices,yields,land_costs,sw_costs,gw_costs,water_nirs,alphas_land,alphas_sw)]
linear_term_sum_total = [p*y - c - aland for p,y,c,aland in zip(prices,yields,land_costs,alphas_total)]
linear_term_sum_sw = [-(swc*n*1000) for swc,n in zip(sw_costs,water_nirs)]  # multiply by 1000 to get to non-corrected NIRs
linear_term_sum_gw = [-(gwc*n*1000) for gwc,n in zip(gw_costs,water_nirs)]
# linear_term_sum_gw = [p*y - c - (gwc*n) - aland for p,y,c,gwc,n,aland in zip(prices,yields,land_costs,gw_costs,water_nirs,alphas_total)]

## B.3. Preparing model vars and params: (these need to be dict()!)
gammas_total=dict(data_profit["gamma"]) #JY temporarily set at 100
# gammas_sw=dict(data_profit["gamma"].head(53835)) #JY temporarily set at 100
net_prices_total=dict(enumerate(linear_term_sum_total))
net_prices_sw=dict(enumerate(linear_term_sum_sw))
net_prices_gw=dict(enumerate(linear_term_sum_gw))
x_start_values=dict(enumerate([0.0]*3))
nirs=dict(water_nirs)

farm_no = 37455

## C.1. Constructing model inputs:
fwm = ConcreteModel()
# fwm.ids = Set(initialize=ids)
fwm.ids = Set(initialize=crop_ids_by_farm[farm_no])
# fwm.farm_ids = Set(initialize=farm_ids)
fwm.farm_ids = Set(initialize=range(farm_no,farm_no + 1))
fwm.crop_ids_by_farm = Set(fwm.farm_ids, initialize=crop_ids_by_farm[farm_no])
fwm.crop_ids_by_farm_and_constraint = Set(fwm.farm_ids, initialize=crop_ids_by_farm_and_constraint[farm_no])
keys_to_extract = crop_ids_by_farm[farm_no]
net_prices_total_subset = {key: net_prices_total[key] for key in keys_to_extract}
net_prices_sw_subset = {key: net_prices_sw[key] for key in keys_to_extract}
net_prices_gw_subset = {key: net_prices_gw[key] for key in keys_to_extract}
gammas_total_subset = {key: gammas_total[key] for key in keys_to_extract}
nirs_subset = {key: nirs[key] for key in keys_to_extract}
fwm.net_prices_total = Param(fwm.ids, initialize=net_prices_total_subset, mutable=True)
fwm.net_prices_sw = Param(fwm.ids, initialize=net_prices_sw_subset, mutable=True)
fwm.net_prices_gw = Param(fwm.ids, initialize=net_prices_gw_subset, mutable=True)
fwm.gammas_total = Param(fwm.ids, initialize=gammas_total_subset, mutable=True)
# fwm.alphas_sw = Param(fwm.farm_ids, initialize=alphas_sw[0], mutable=True)
# fwm.gammas_sw = Param(fwm.farm_ids, initialize=gammas_sw[0], mutable=True)
fwm.land_constraints = Param(fwm.farm_ids, initialize=land_constraints_by_farm[farm_no], mutable=True)
fwm.sw_constraints = Param(fwm.farm_ids, initialize=sw_constraints_by_farm[farm_no], mutable=True)
fwm.gw_constraints = Param(fwm.farm_ids, initialize=gw_constraints_by_farm[farm_no], mutable=True)
# fwm.water_constraints = Param(fwm.farm_ids, initialize=water_constraints_by_farm[0], mutable=True)
fwm.xs_total = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)
fwm.xs_sw = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)
fwm.xs_gw = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)
obs_lu_total = dict(data_profit["area_irrigated"])
obs_lu_total_subset = {key: obs_lu_total[key] for key in keys_to_extract}
# obs_lu_total_subset[0] = .001
# obs_lu_total_subset[107670] = .001
# obs_lu_total_subset[161505] = .001
# obs_lu_total_subset[215340] = .001
# obs_lu_total_subset[269175] = .001
# obs_lu_total_subset[376845] = .001
# obs_lu_sw = dict(area_irrigated_sw_farm["area_irrigated_sw"])
fwm.obs_lu_total_subset = Param(fwm.ids, initialize=obs_lu_total_subset, mutable=True)
# fwm.obs_lu_sw_subset = Param(fwm.farm_ids, initialize=obs_lu_sw[0] + 0.006, mutable=True)  # JY EDIT
fwm.nirs = Param(fwm.ids, initialize=nirs_subset, mutable=True)

## C.2. Constructing model functions:
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
    return fwm.xs_total[i] == fwm.obs_lu_total_subset[i]
fwm.c3 = Constraint(fwm.ids, rule=obs_lu_constraint_total)

# def obs_lu_constraint_sw(fwm, f):
    # return sum(fwm.xs_sw[i] for i in fwm.crop_ids_by_farm[f]) == fwm.obs_lu_sw_subset[f]
# fwm.c4 = Constraint(fwm.farm_ids, rule=obs_lu_constraint_sw)

# def obs_lu_constraint_sw(fwm, i):
#     return fwm.xs_sw[i] == fwm.obs_lu_sw[i]
# fwm.c4 = Constraint(fwm.ids, rule=obs_lu_constraint_sw)

# def water_constraint(fwm, ff):
#     return sum(fwm.xs_sw[i]*fwm.nirs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff]) <= fwm.water_constraints[ff]
# fwm.c2 = Constraint(fwm.farm_ids, rule=water_constraint)

def sw_constraint(fwm, ff):
    return sum(fwm.xs_sw[i]*fwm.nirs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff]) <= fwm.sw_constraints[ff]
fwm.c2 = Constraint(fwm.farm_ids, rule=sw_constraint)

def gw_constraint(fwm, ff):
    return sum(fwm.xs_gw[i]*fwm.nirs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff]) <= fwm.gw_constraints[ff]
fwm.c6 = Constraint(fwm.farm_ids, rule=gw_constraint)

def obs_lu_constraint_sum(fwm, i):
    return fwm.xs_sw[i] + fwm.xs_gw[i] == fwm.xs_total[i]
fwm.c5 = Constraint(fwm.ids, rule=obs_lu_constraint_sum)

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

# obs_lu_duals_sw = dict()
# for c in fwm.component_objects(Constraint, active=True):
#     if str(c) == "c4":
#         cobject = getattr(fwm, str(c))
#         for index in cobject:
#             obs_lu_duals_sw[index] = fwm.dual[cobject[index]]

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
gamma1_total = [((2. * a / b) if (b > 0.0) else 0.0) for a, b in zip(obs_lu_duals_total.values(), obs_lu_total_subset.values())]
alpha1_total = [-(0.5 * a * b) for a, b in zip(gamma1_total, obs_lu_total_subset.values())]
print(alpha1_total)
print("+++ alpha check: +++")
print([(a, b) for a, b in zip(alphas_total, alpha1_total)])
print("+++ gamma check: +++")
print([(a, b) for a, b in zip(gammas_total, gamma1_total)])
alphas_total = alpha1_total


keys_to_extract = crop_ids_by_farm[farm_no]
prices_subset = {key: prices[key] for key in keys_to_extract}
yields_subset = {key: yields[key] for key in keys_to_extract}
land_costs_subset = {key: land_costs[key] for key in keys_to_extract}
linear_term_sum_total = [p*y - c - aland for p,y,c,aland in zip(prices_subset.values(),yields_subset.values(),land_costs_subset.values(),alphas_total)]
gammas_total = dict(enumerate(gamma1_total))
net_prices_total = dict(enumerate(linear_term_sum_total))


net_prices_total[keys_to_extract[0]] = net_prices_total.pop(0)
net_prices_total[keys_to_extract[1]] = net_prices_total.pop(1)
net_prices_total[keys_to_extract[2]] = net_prices_total.pop(2)
net_prices_total[keys_to_extract[3]] = net_prices_total.pop(3)
net_prices_total[keys_to_extract[4]] = net_prices_total.pop(4)
net_prices_total[keys_to_extract[5]] = net_prices_total.pop(5)
net_prices_total[keys_to_extract[6]] = net_prices_total.pop(6)
net_prices_total[keys_to_extract[7]] = net_prices_total.pop(7)
net_prices_total[keys_to_extract[8]] = net_prices_total.pop(8)
net_prices_total[keys_to_extract[9]] = net_prices_total.pop(9)
gammas_total[keys_to_extract[0]] = gammas_total.pop(0)
gammas_total[keys_to_extract[1]] = gammas_total.pop(1)
gammas_total[keys_to_extract[2]] = gammas_total.pop(2)
gammas_total[keys_to_extract[3]] = gammas_total.pop(3)
gammas_total[keys_to_extract[4]] = gammas_total.pop(4)
gammas_total[keys_to_extract[5]] = gammas_total.pop(5)
gammas_total[keys_to_extract[6]] = gammas_total.pop(6)
gammas_total[keys_to_extract[7]] = gammas_total.pop(7)
gammas_total[keys_to_extract[8]] = gammas_total.pop(8)
gammas_total[keys_to_extract[9]] = gammas_total.pop(9)
#
# gamma1_sw = [((2. * a / b) if (b > 0.0) else 0.0) for a, b in zip(obs_lu_duals_sw.values(), [obs_lu_sw[0]])]
# alpha1_sw = [-(0.5 * a * b) for a, b in zip(gamma1_sw, [obs_lu_sw[0]])]
# print(alpha1_sw)
# print("+++ alpha check: +++")
# print([(a, b) for a, b in zip(alphas_sw, alpha1_sw)])
# print("+++ gamma check: +++")
# print([(a, b) for a, b in zip(gammas_sw, gamma1_sw)])
# alphas_sw = dict(enumerate(alpha1_sw))
# gammas_sw = dict(enumerate(gamma1_sw))

# # TEMP
# import csv
# with open('gammas_sw.csv', 'w') as f:
#     writer = csv.writer(f, lineterminator='\n')
#     for key,val in gammas_sw.items():
#         writer.writerow([val])

## C.2. 2st stage: Quadratic model included in JWP model simulations
## C.2.a. Constructing model inputs:
##  (repetition to be safe - deepcopy does not work on PYOMO models)
## C.1. Constructing model inputs:
fwm_s = ConcreteModel()
# fwm_s.ids = Set(initialize=ids)
fwm_s.ids = Set(initialize=crop_ids_by_farm[farm_no])
# fwm_s.farm_ids = Set(initialize=farm_ids[0]) # JY temp, index [n] indicates number of farms (n+1) to include in chunk
fwm_s.farm_ids = Set(initialize=range(farm_no, farm_no + 1))
fwm_s.crop_ids_by_farm = Set(fwm_s.farm_ids, initialize=crop_ids_by_farm[farm_no])
fwm_s.crop_ids_by_farm_and_constraint = Set(fwm_s.farm_ids, initialize=crop_ids_by_farm_and_constraint[farm_no])
keys_to_extract = crop_ids_by_farm[farm_no]
net_prices_total_subset = {key: net_prices_total[key] for key in keys_to_extract}
net_prices_sw_subset = {key: net_prices_sw[key] for key in keys_to_extract}
net_prices_gw_subset = {key: net_prices_gw[key] for key in keys_to_extract}
gammas_total_subset = {key: gammas_total[key] for key in keys_to_extract}
nirs_subset = {key: nirs[key] for key in keys_to_extract}

# !JY! replace net_prices with zero value for gammas that equal to zero
for key,val in net_prices_total_subset.items():
    if gammas_total_subset[key] == 0:
        net_prices_total_subset[key] = 0
        # net_prices_sw_subset[key] = -999999999 # JY where crop is unobserved, set costs to artificially high value so crop is excluded
        # net_prices_gw_subset[key] = -999999999 # JY where crop is unobserved, set costs to artificially high value so crop is excluded

fwm_s.net_prices_total = Param(fwm_s.ids, initialize=net_prices_total_subset, mutable=True)
fwm_s.net_prices_sw = Param(fwm_s.ids, initialize=net_prices_sw_subset, mutable=True)

fwm_s.net_prices_gw = Param(fwm_s.ids, initialize=net_prices_gw_subset, mutable=True)
fwm_s.gammas_total = Param(fwm_s.ids, initialize=gammas_total_subset, mutable=True)
# fwm_s.alphas_sw = Param(fwm_s.farm_ids, initialize=alphas_sw[0], mutable=True)
# fwm_s.gammas_sw = Param(fwm_s.farm_ids, initialize=gammas_sw[0], mutable=True)
fwm_s.land_constraints = Param(fwm_s.farm_ids, initialize=land_constraints_by_farm[farm_no], mutable=True)
# fwm_s.water_constraints = Param(fwm_s.farm_ids, initialize=water_constraints_by_farm[0], mutable=True)
sw_constraints_by_farm_rev = sw_constraints_by_farm.copy()
sw_constraints_by_farm_rev.update((x, y*0.8) for x, y in sw_constraints_by_farm_rev.items())
fwm_s.sw_constraints = Param(fwm_s.farm_ids, initialize=sw_constraints_by_farm_rev[farm_no], mutable=True)
with open('data_inputs/pickles/sw_gw_constr_20220321/gw_calib_constraints_20220401_protocol2.p', 'rb') as fp:
    gw_constraints_by_farm = pickle.load(fp, encoding='latin1')
fwm_s.gw_constraints = Param(fwm_s.farm_ids, initialize=gw_constraints_by_farm[farm_no], mutable=True)
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
fwm_s.obj_f = Objective(rule=obj_fun, sense=maximize)

def land_constraint(fwm_s, ff):
    return sum(fwm_s.xs_total[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.land_constraints[ff]
fwm_s.c1 = Constraint(fwm_s.farm_ids, rule=land_constraint)

def obs_lu_constraint_sum(fwm_s, i):
    return fwm_s.xs_sw[i] + fwm_s.xs_gw[i] == fwm_s.xs_total[i]
fwm_s.c5 = Constraint(fwm_s.ids, rule=obs_lu_constraint_sum)
#
def sw_constraint(fwm_s, ff):
    return sum(fwm_s.xs_sw[i]*fwm_s.nirs[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.sw_constraints[ff]
fwm_s.c2 = Constraint(fwm_s.farm_ids, rule=sw_constraint)

def gw_constraint(fwm_s, ff):
    return sum(fwm_s.xs_gw[i]*fwm_s.nirs[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.gw_constraints[ff]
fwm_s.c6 = Constraint(fwm_s.farm_ids, rule=gw_constraint)

# def water_constraint(fwm_s, ff):
#     return sum(fwm_s.xs[i]*fwm_s.nirs[i] for i in fwm_s.crop_ids_by_farm_and_constraint[ff]) <= fwm_s.water_constraints[ff]
# fwm_s.c2 = Constraint(fwm_s.farm_ids, rule=water_constraint)

## C.2.c Creating and running the solver:
import datetime
start_time = datetime.datetime.now()
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

# JY store results into a pandas dataframe
results_pd = data_profit
results_pd = results_pd.assign(calc_area=result_xs.values())
results_pd = results_pd.assign(nir=nirs.values())
results_pd['calc_water_demand'] = results_pd['calc_area'] * results_pd['nir']
results_pivot = pd.pivot_table(results_pd, index=['nldas'], values=['calc_water_demand'], aggfunc=np.sum)