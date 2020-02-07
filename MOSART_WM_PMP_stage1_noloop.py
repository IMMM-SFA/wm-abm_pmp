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
crop_ids_by_farm_and_constraint={}
land_constraints_by_farm={}
water_constraints_by_farm={}
#crop_ids_by_farm=dict(enumerate([np.where(data_profit["nldas"]==nldas_ids[i])[0].tolist() for i in range(53835)])) #JY this takes forever, find better way
with open('crop_ids_by_farm.p', 'rb') as fp:
    crop_ids_by_farm = pickle.load(fp)
with open('land_constraints_by_farm.p', 'rb') as fp:
    land_constraints_by_farm = pickle.load(fp)
with open('water_constraints_by_farm_v2.p', 'rb') as fp:
    water_constraints_by_farm = pickle.load(fp)
with open('crop_ids_by_farm_and_constraint.p', 'rb') as fp:
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
fwm = ConcreteModel()
fwm.ids = Set(initialize=ids)
fwm.farm_ids = Set(initialize=farm_ids)
fwm.crop_ids_by_farm = Set(fwm.farm_ids, initialize=crop_ids_by_farm)
fwm.crop_ids_by_farm_and_constraint = Set(fwm.farm_ids, initialize=crop_ids_by_farm_and_constraint)
fwm.net_prices = Param(fwm.ids, initialize=net_prices, mutable=True)
fwm.gammas = Param(fwm.ids, initialize=gammas, mutable=True)
fwm.land_constraints = Param(fwm.farm_ids, initialize=land_constraints_by_farm, mutable=True)
fwm.water_constraints = Param(fwm.farm_ids, initialize=water_constraints_by_farm, mutable=True) #JY here need to read calculate new water constraints
fwm.xs = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)
fwm.nirs = Param(fwm.ids, initialize=nirs, mutable=True)

## C.2.b. 2nd stage model: Constructing functions:
def obj_fun(fwm):
    return 0.00001*sum(sum((fwm.net_prices[i] * fwm.xs[i] - 0.5 * fwm.gammas[i] * fwm.xs[i] * fwm.xs[i]) for i in fwm.crop_ids_by_farm[f]) for f in fwm.farm_ids)
fwm.obj_f = Objective(rule=obj_fun, sense=maximize)

def land_constraint(fwm, ff, su, rf):
    return sum(fwm.xs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff,su,rf]) <= fwm.land_constraints[ff,su,rf]
fwm.c1 = Constraint(fwm.farm_ids, fwm.binary_ids, fwm.binary_ids, rule=land_constraint)

def water_constraint(fwm, ff, su, rf):
    return sum(fwm.xs[i]*fwm.nirs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff,su,rf]) <= fwm.water_constraints[ff,su,rf]
fwm.c2 = Constraint(fwm.farm_ids, fwm.binary_ids, fwm.binary_ids, rule=water_constraint)
