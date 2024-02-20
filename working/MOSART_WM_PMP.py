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


data_file=pd.ExcelFile("MOSART_WM_PMP_inputs_test.xlsx")
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
with open('../data_inputs/crop_ids_by_farm.p', 'rb') as fp:
    crop_ids_by_farm = pickle.load(fp)
with open('../data_inputs/land_constraints_by_farm.p', 'rb') as fp:
    land_constraints_by_farm = pickle.load(fp)
with open('../archived/water_constraints_by_farm.p', 'rb') as fp:
    water_constraints_by_farm = pickle.load(fp)
with open('../data_inputs/crop_ids_by_farm_and_constraint.p', 'rb') as fp:
    crop_ids_by_farm_and_constraint = pickle.load(fp)

#JY this outputs pandas series for each entry of land_constraints_by_farm and water_constraints_by_farm; need
#to convert to float
for i in range(53835):
    #crop_ids_by_farm_and_constraint[i]=np.where((data_profit["nldas"]==nldas_ids[i]))[0].tolist()
    constraint_ids=np.where((data_constraint["nldas"]==nldas_ids[i]))[0]
    land_constraints_by_farm[i]=data_constraint.iloc[constraint_ids]["land_constraint"].astype('float')
    #water_constraints_by_farm[i]=data_constraint.iloc[constraint_ids]["water_constraint"].astype('float')


## B.2. Preparing linear profit coefficients:
prices=data_profit["price"]
yields=data_profit["yield"]
land_costs=data_profit["land_cost"]
water_costs=data_profit["sw_cost"] #JY need to fill in gaps for DC, figure out how to handle groundwater costs
alphas=data_profit["alpha"] #JY temporarily set as -100

linear_term_sum=[p*y - c - wc - a for p,y,c,wc,a in zip(prices,yields,land_costs,water_costs,alphas)]

## B.3. Preparing model vars and params: (these need to be dict()!)
gammas=dict(data_profit["gamma"]) #JY temporarily set at 100
net_prices=dict(enumerate(linear_term_sum))
x_start_values=dict(enumerate([0.0]*3))

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

## C.2. Constructing model functions:
def obj_fun(fwm):
    return 0.00001*sum(sum((fwm.net_prices[i] * fwm.xs[i] - 0.5 * fwm.gammas[i] * fwm.xs[i] * fwm.xs[i]) for i in fwm.crop_ids_by_farm[f]) for f in fwm.farm_ids)
fwm.obj_f = Objective(rule=obj_fun, sense=maximize)

def land_constraint(fwm, ff):
    return sum(fwm.xs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff]) <= fwm.land_constraints[ff]
fwm.c1 = Constraint(fwm.farm_ids, rule=land_constraint)

# def water_constraint(fwm, ff):
#     return sum(fwm.xs[i]*fwm.nirs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff]) <= fwm.water_constraints[ff]
# fwm.c2 = Constraint(fwm.farm_ids, rule=water_constraint)

## C.3. Creating and running the solver:
opt = SolverFactory("ipopt", solver_io='nl')
results = opt.solve(fwm, keepfiles=False, tee=True)
print(results.solver.termination_condition)

