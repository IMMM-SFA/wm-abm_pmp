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


data_file=pd.ExcelFile("MOSART_WM_PMP_inputs_test.xlsx")
data_profit = data_file.parse("Profit")
data_constraint = data_file.parse("Constraint")
water_nirs=data_profit["nir_corrected"]
nirs=dict(water_nirs)

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
# for i in range(53835):
    #crop_ids_by_farm_and_constraint[i]=np.where((data_profit["nldas"]==nldas_ids[i]))[0].tolist()
    #constraint_ids=np.where((data_constraint["nldas"]==nldas_ids[i]))[0]
    #land_constraints_by_farm[i]=data_constraint.iloc[constraint_ids]["land_constraint"].astype('float')
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
obs_lu = dict(data_profit["area_irrigated"])

chunk = 1000
no_chunks = 53
counter = 0
## C.3. Loop through farms and solve
for c in range(590): # !JY! replace range(10) with farm_ids
    ids_data = []
    for f in range(counter*1000, (counter+1)*1000):
        ids_data = ids_data + crop_ids_by_farm[f]
    ids_data.sort()
    farm_ids_data = range(counter*1000, (counter+1)*1000)
    crop_ids_by_farm_data = dict((k, crop_ids_by_farm[k]) for k in range(counter*1000, (counter+1)*1000))
    crop_ids_by_farm_and_constraint_data = dict((k, crop_ids_by_farm[k]) for k in range(counter*1000, (counter+1)*1000))
    net_prices_data = dict((k, net_prices[k]) for k in ids_data)
    gammas_data = dict((k, gammas[k]) for k in ids_data)
    obs_lu_data = dict((k, obs_lu[k]) for k in ids_data)
    land_constraints_by_farm_data = dict((k, land_constraints_by_farm[k]) for k in range(counter*1000, (counter+1)*1000))

    ## C.1. Constructing model inputs:
    fwm = ConcreteModel()
    fwm.ids = Set(initialize=ids_data)
    fwm.farm_ids = Set(initialize=farm_ids_data)
    fwm.crop_ids_by_farm = Set(fwm.farm_ids, initialize=crop_ids_by_farm_data)
    fwm.crop_ids_by_farm_and_constraint = Set(fwm.farm_ids, initialize=crop_ids_by_farm_and_constraint_data)
    fwm.net_prices = Param(fwm.ids, initialize=net_prices_data, mutable=True)
    fwm.gammas = Param(fwm.ids, initialize=gammas_data, mutable=True)
    fwm.land_constraints = Param(fwm.farm_ids, initialize=land_constraints_by_farm_data, mutable=True)
    #fwm.water_constraints = Param(fwm.farm_ids, initialize=water_constraints_by_farm, mutable=True)
    fwm.xs = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)
    obs_lu = dict(data_profit["area_irrigated"])
    fwm.obs_lu = Param(fwm.ids, initialize=obs_lu_data, mutable=True)

    ## C.2. Constructing model functions:
    def obj_fun(fwm):
        return sum(sum((fwm.net_prices[i] * fwm.xs[i]) for i in fwm.crop_ids_by_farm[f]) for f in fwm.farm_ids)
    fwm.obj_f = Objective(rule=obj_fun, sense=maximize)

    def land_constraint(fwm, ff):
        return sum(fwm.xs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff]) <= fwm.land_constraints[ff]
    fwm.c1 = Constraint(fwm.farm_ids, rule=land_constraint)


    def obs_lu_constraint(fwm, i):
        return fwm.xs[i] == fwm.obs_lu[i]
    fwm.c3 = Constraint(fwm.ids, rule=obs_lu_constraint)

    fwm.dual = Suffix(direction=Suffix.IMPORT)

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
    linear_term_sum = [p * y - c - wc - a for p, y, c, wc, a in zip(prices, yields, land_costs, water_costs, alphas)]
    gammas = dict(enumerate(gamma1))
    net_prices = dict(enumerate(linear_term_sum))

    counter += 1

    pdb.set_trace()
