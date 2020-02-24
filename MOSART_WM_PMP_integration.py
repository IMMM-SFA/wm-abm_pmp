from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

## C.1. Preparing model indices and constraints:
ids = range(592185) # total number of crop and nldas ID combinations
farm_ids = range(53835) # total number of farm agents / nldas IDs
with open('crop_ids_by_farm.p', 'rb') as fp:
    crop_ids_by_farm = pickle.load(fp)
with open('crop_ids_by_farm_and_constraint.p', 'rb') as fp:
    crop_ids_by_farm_and_constraint = pickle.load(fp)

# Load gammas and alphas
with open('gammas.p', 'rb') as fp:
    gammas = pickle.load(fp)
with open('net_prices.p', 'rb') as fp:
    net_prices = pickle.load(fp)


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