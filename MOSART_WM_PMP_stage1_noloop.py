import os
import sys
global basepath
print(os.path.dirname(sys.argv[0]))
##basepath = os.path.dirname(sys.argv[0]).split(__file__)[0]
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
import pdb


data_file=pd.ExcelFile("MOSART_WM_PMP_inputs_v1.xlsx")
data_profit = data_file.parse("Profit")
data_constraint = data_file.parse("Constraint")

nldas_ids=data_profit["nldas"][0:53835].tolist()

## B.1. Preparing model indices and constraints:
ids = range(592185) # total number of crop and nldas ID combinations
farm_ids = range(53835) # total number of farm agents / nldas IDs
sd_no = len(farm_ids)
crop_types=[str(i) for i in list(pd.unique(data_profit["crop"]))]
crop_no=len(crop_types)
crop_ids_by_farm_and_constraint={}
land_constraints_by_farm={}
water_constraints_by_farm={}
#crop_ids_by_farm=dict(enumerate([np.where(data_profit["nldas"]==nldas_ids[i])[0].tolist() for i in range(53835)])) #JY this takes forever, find better way
with open('data_inputs/crop_ids_by_farm.p', 'rb') as fp:
    crop_ids_by_farm = pickle.load(fp)
with open('data_inputs/land_constraints_by_farm.p', 'rb') as fp:
    land_constraints_by_farm = pickle.load(fp)
with open('data_inputs/water_constraints_by_farm_v2.p', 'rb') as fp:
    water_constraints_by_farm = pickle.load(fp)
with open('data_inputs/crop_ids_by_farm_and_constraint.p', 'rb') as fp:
    crop_ids_by_farm_and_constraint = pickle.load(fp)

water_constraints_by_farm = dict.fromkeys(water_constraints_by_farm, 9999999999)

# JY this outputs pandas series for each entry of land_constraints_by_farm and water_constraints_by_farm; need
# to convert to float
# for i in range(53835):
#     #crop_ids_by_farm_and_constraint[i]=np.where((data_profit["nldas"]==nldas_ids[i]))[0].tolist()
#     constraint_ids=np.where((data_constraint["nldas"]==nldas_ids[i]))[0]
#     #land_constraints_by_farm[i]=data_constraint.iloc[constraint_ids]["land_constraint"].astype('float')
#     water_constraints_by_farm_2[i]=data_constraint.iloc[constraint_ids]["water_constraint"].astype('float')


## B.2. Preparing linear profit coefficients:
prices=data_profit["price"]
yields=data_profit["yield"]
land_costs=data_profit["land_cost"]
water_nirs=data_profit["nir_corrected"]
water_costs=data_profit["sw_cost"] #JY need to fill in gaps for DC, figure out how to handle groundwater costs
alphas=data_profit["alpha"] #JY temporarily set as -100

linear_term_sum=[p*y - c - wc*n - a for p,y,c,wc,n,a in zip(prices,yields,land_costs,water_costs,water_nirs,alphas)]

## B.3. Preparing model vars and params: (these need to be dict()!)
gammas=dict(data_profit["gamma"]) #JY temporarily set at 100
net_prices=dict(enumerate(linear_term_sum))
x_start_values=dict(enumerate([0.0]*3))
nirs=dict(water_nirs)


## C.1. Constructing model inputs:
fwm = ConcreteModel()
fwm.ids = Set(initialize=ids)
fwm.farm_ids = Set(initialize=farm_ids)
fwm.crop_ids_by_farm = Set(fwm.farm_ids, initialize=crop_ids_by_farm)
fwm.crop_ids_by_farm_and_constraint = Set(fwm.farm_ids, initialize=crop_ids_by_farm_and_constraint)
fwm.net_prices = Param(fwm.ids, initialize=net_prices, mutable=True)
fwm.gammas = Param(fwm.ids, initialize=gammas, mutable=True)
fwm.land_constraints = Param(fwm.farm_ids, initialize=land_constraints_by_farm, mutable=True)
fwm.water_constraints = Param(fwm.farm_ids, initialize=water_constraints_by_farm, mutable=True)
fwm.xs = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)
obs_lu = dict(data_profit["area_irrigated"])
fwm.obs_lu = Param(fwm.ids, initialize=obs_lu, mutable=True)
fwm.nirs = Param(fwm.ids, initialize=nirs, mutable=True)

## C.2. Constructing model functions:
def obj_fun(fwm):
    return sum(sum((fwm.net_prices[i] * fwm.xs[i]) for i in fwm.crop_ids_by_farm[f]) for f in fwm.farm_ids)
fwm.obj_f = Objective(rule=obj_fun, sense=maximize)

# JY need to re-implement this
# def land_constraint(fwm, ff):
#     return sum(fwm.xs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff]) <= fwm.land_constraints[ff]
# fwm.c1 = Constraint(fwm.farm_ids, rule=land_constraint)


def obs_lu_constraint(fwm, i):
    return fwm.xs[i] == fwm.obs_lu[i]
fwm.c3 = Constraint(fwm.ids, rule=obs_lu_constraint)

fwm.dual = Suffix(direction=Suffix.IMPORT)

def water_constraint(fwm, ff):
    return sum(fwm.xs[i]*fwm.nirs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff]) <= fwm.water_constraints[ff]
fwm.c2 = Constraint(fwm.farm_ids, rule=water_constraint)

## C.3. Solve
opt = SolverFactory("ipopt", solver_io='nl')
results = opt.solve(fwm, keepfiles=False, tee=True)
print(results.solver.termination_condition)

## C.1.d. Save duals:
from pyomo.core import Constraint

obs_lu_duals = dict()
for c in fwm.component_objects(Constraint, active=True):
    if str(c) == "c3":
        cobject = getattr(fwm, str(c))
        for index in cobject:
            obs_lu_duals[index] = fwm.dual[cobject[index]]

## C.1.e. 1st stage result: Calculate alpha and gamma:
# gamma1 = [((2. * a / b) if (b > 0.0) else 0.0) for a,b in zip(obs_lu_duals.values(),obs_lu.values())]
gamma1 = [((a / b) if (b > 0.0) else 0.0) for a, b in zip(obs_lu_duals.values(), obs_lu.values())]
alpha1 = [-(0.5 * a * b) for a, b in zip(gamma1, obs_lu.values())]
print(alpha1)
print("+++ alpha check: +++")
print([(a, b) for a, b in zip(alphas, alpha1)])
print("+++ gamma check: +++")
print([(a, b) for a, b in zip(gammas, gamma1)])
alphas = alpha1
linear_term_sum=[p*y - c - wc*n - a for p,y,c,wc,n,a in zip(prices,yields,land_costs,water_costs,water_nirs,alphas)]
gammas = dict(enumerate(gamma1))
net_prices = dict(enumerate(linear_term_sum))

## C.2. 2st stage: Quadratic model included in JWP model simulations
## C.2.a. Constructing model inputs:
##  (repetition to be safe - deepcopy does not work on PYOMO models)
fwm_s = ConcreteModel()
fwm_s.ids = Set(initialize=ids)
fwm_s.farm_ids = Set(initialize=farm_ids)
fwm_s.crop_ids_by_farm = Set(fwm.farm_ids, initialize=crop_ids_by_farm)
fwm_s.crop_ids_by_farm_and_constraint = Set(fwm.farm_ids, initialize=crop_ids_by_farm_and_constraint)
fwm_s.net_prices = Param(fwm.ids, initialize=net_prices, mutable=True)
fwm_s.gammas = Param(fwm.ids, initialize=gammas, mutable=True)
fwm_s.land_constraints = Param(fwm.farm_ids, initialize=land_constraints_by_farm, mutable=True)
fwm_s.water_constraints = Param(fwm.farm_ids, initialize=water_constraints_by_farm, mutable=True) #JY here need to read calculate new water constraints
fwm_s.xs = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)
fwm_s.nirs = Param(fwm.ids, initialize=nirs, mutable=True)

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

# JY results stored as pickle file (results_xs.p). Start here and load pickle files.
with open('result_xs.p', 'rb') as fp:
    result_xs = pickle.load(fp)

# JY store results into a pandas dataframe
results_pd = data_profit
results_pd = results_pd.assign(calc_area=result_xs.values())
results_pd = results_pd.assign(nir=nirs.values())
results_pd['calc_water_demand'] = results_pd['calc_area'] * results_pd['nir']
results_pivot = pd.pivot_table(results_pd, index=['nldas'], values=['calc_water_demand'], aggfunc=np.sum)


# ## D.2. Storing derived model outputs:
# df_output=pd.DataFrame(index=range(0,(sd_no*crop_no)),columns=["subdistrict","crop","land_use","irrigation","profit",
#                                                                  "alpha","gamma","land_ratio_to_obs"])
# df_output_mth=pd.DataFrame(index=range(1,1+(sd_no*12*crop_no)),columns=["subdistrict","month","crop","land_use","irrigation","profit",
#                                                                         "alpha","gamma"])

# #JY this look takes forever! Replace!
# i=1
# for sd in range(sd_no):
#     for crp in range(crop_no):
#         crop_id=crop_ids_by_farm[sd][crp]
#         current_land_use=result_xs[crop_id] if result_xs[crop_id] else 0.0
#         current_water_use=current_land_use*water_nirs[crop_id]
#         current_profit=current_land_use*(net_prices[crop_id]+alphas[crop_id])
#         current_alpha1=alpha1[crop_id]
#         current_gamma1=gamma1[crop_id]
#         land_ratio_to_obs = (current_land_use / obs_lu[crop_id]) if (obs_lu[crop_id] > 0.0) else int(obs_lu[crop_id] == round(current_land_use,0))
#         df_output.loc[crop_id]=[farm_ids[sd],crop_types[crp],current_land_use,current_water_use,current_profit,
#                                 current_alpha1,current_gamma1,land_ratio_to_obs]
#         for mth in range(1,13):
#             #water_percent=float(data_seasons["water_percent"][(data_seasons["crop"]==crop_types[crp])&(data_seasons["month"]==mth)]) ##JY identify monthly variation of water use by crop
#             current_water_use_mth=current_water_use / 12.0
#             df_output_mth.loc[i]=[farm_ids[sd],mth,crop_types[crp],current_land_use,current_water_use_mth,current_profit,
#                                   current_alpha1,current_gamma1]
#             i+=1