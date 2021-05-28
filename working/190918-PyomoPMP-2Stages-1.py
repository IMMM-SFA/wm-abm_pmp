import os
import sys
global basepath
print(os.path.dirname(sys.argv[0]))
#basepath = os.path.dirname(sys.argv[0]).split(__file__)[0]
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np

## A.1. Loading main model data:
run_mode = int(input("Enter \"1\" for using PMP alpha and gamma values from the 1st stage calculated her, \"0\" for exogenous values:"))
print("Run mode is: " + ("endogenous" if (run_mode == 1) else "exogenous") + " PMP alpha and gamma.")
filename = "190918-PyomoPMP-2Stages-Inputs-1.xlsx"
print("Input selected: " + filename)
#filename_and_path=os.path.join(basepath, filename)
data_file=pd.ExcelFile('190918-PyomoPMP-2Stages-Inputs-1.xlsx')
data_profit = data_file.parse("Profit")
data_constraints = data_file.parse("Constraints")
#crop_types=["Barley","OtherVegW","OtherFld","Olive","OtherTrees","OtherVegS"]
crop_types=[str(i) for i in list(pd.unique(data_profit["crop"]))]
subdistricts=data_profit["subdistrict"][0:89].tolist()
crop_no=len(crop_types)
sd_no=len(subdistricts)

## A.2. Loading post-processing data, etc.:
data_seasons = data_file.parse("iCWDMNTH")
data_seasons = data_file.parse("iCWDMNTH")
gw_nirs=dict(data_profit["gw_nir"])


## B.1. Preparing model indices and constraints:
ids=range(534)
farm_ids=range(89)
binary_ids=range(2)
crop_ids_by_farm=dict(enumerate([np.where(data_profit["subdistrict"]==subdistricts[i])[0].tolist() for i in range(89)]))
crop_ids_by_farm_and_constraint={}
land_constraints_by_farm={}
water_constraints_by_farm={}
for i in range(89):
    for su in range(2):
        for rf in range(2):
            crop_ids_by_farm_and_constraint[i,su,rf]=np.where((data_profit["subdistrict"]==subdistricts[i])&
                                                              (data_profit["summer"]==su)&
                                                              (data_profit["rainfed"]==rf))[0].tolist()
            constraint_ids=np.where((data_constraints["subdistrict"]==subdistricts[i])&
                                    (data_constraints["summer"]==su)&
                                    (data_constraints["rainfed"]==rf))[0]
            land_constraints_by_farm[i,su,rf]=float(data_constraints.iloc[constraint_ids]["land_constraint"])
            water_constraints_by_farm[i,su,rf]=float(data_constraints.iloc[constraint_ids]["water_constraint"])

## B.2. Preparing linear profit coefficients:
prices=data_profit["price"] # dict()
yields=data_profit["yield"]
land_costs=data_profit["land_cost"]
water_pcfs=data_profit["water_pcf"]
water_nirs=data_profit["water_nir"]
alphas=data_profit["alpha"]
water_pheads=data_profit["water_phead"]
## Calculating derived values:
linear_term_sum=[p*y - c - h*f*n - a for p,y,c,h,f,n,a in zip(prices,yields,land_costs,water_pheads,water_pcfs,water_nirs,alphas)]

## B.3. Preparing model vars and params: (these need to be dict()!)
gammas=dict(data_profit["gamma"])
nirs=dict(water_nirs)
net_prices=dict(enumerate(linear_term_sum))
x_start_values=dict(enumerate([0.0]*3))


## C.1. 1st stage: Linear model to get calibration coefs (alpha & gamma)  
## C.1.a. Constructing model inputs:
fwmL = ConcreteModel()
fwmL.ids = Set(initialize=ids)
fwmL.farm_ids = Set(initialize=farm_ids)
fwmL.binary_ids = Set(initialize=binary_ids)
fwmL.crop_ids_by_farm = Set(fwmL.farm_ids, initialize=crop_ids_by_farm)
fwmL.crop_ids_by_farm_and_constraint = Set(fwmL.farm_ids, fwmL.binary_ids, fwmL.binary_ids, initialize=crop_ids_by_farm_and_constraint)
fwmL.net_prices = Param(fwmL.ids, initialize=net_prices, mutable=True)
fwmL.gammas = Param(fwmL.ids, initialize=gammas, mutable=True)
fwmL.land_constraints = Param(fwmL.farm_ids, fwmL.binary_ids, fwmL.binary_ids, initialize=land_constraints_by_farm, mutable=True)
fwmL.water_constraints = Param(fwmL.farm_ids, fwmL.binary_ids, fwmL.binary_ids, initialize=water_constraints_by_farm, mutable=True)
fwmL.nirs = Param(fwmL.ids, initialize=nirs, mutable=True)
fwmL.xs = Var(fwmL.ids, domain=NonNegativeReals, initialize=x_start_values)
## Adding variable not in the stage 2 model:
obs_lu=dict(data_profit["observed_land_use"])
fwmL.obs_lu = Param(fwmL.ids, initialize=obs_lu, mutable=True)

## C.1.b. 1st stage model: Constructing functions:
def obj_fun(fwmL):
    return sum(sum((fwmL.net_prices[i] * fwmL.xs[i]) for i in fwmL.crop_ids_by_farm[f]) for f in fwmL.farm_ids)
fwmL.obj_f = Objective(rule=obj_fun, sense=maximize)

def land_constraint(fwmL, ff, su, rf):
    return sum(fwmL.xs[i] for i in fwmL.crop_ids_by_farm_and_constraint[ff,su,rf]) <= fwmL.land_constraints[ff,su,rf]
fwmL.c1 = Constraint(fwmL.farm_ids, fwmL.binary_ids, fwmL.binary_ids, rule=land_constraint)

def water_constraint(fwmL, ff, su, rf):
    return sum(fwmL.xs[i]*fwmL.nirs[i] for i in fwmL.crop_ids_by_farm_and_constraint[ff,su,rf]) <= fwmL.water_constraints[ff,su,rf]
fwmL.c2 = Constraint(fwmL.farm_ids, fwmL.binary_ids, fwmL.binary_ids, rule=water_constraint)

def obs_lu_constraint(fwmL, i):
    return fwmL.xs[i] == fwmL.obs_lu[i]
fwmL.c3 = Constraint(fwmL.ids, rule=obs_lu_constraint)

## Tell the model to record duals ("shadow prices"):
fwmL.dual = Suffix(direction=Suffix.IMPORT)

## C.1.c. Creating and running the solver:
optL = SolverFactory("ipopt", solver_io='nl')
resultsL = optL.solve(fwmL, keepfiles=False, tee=True)
print(resultsL.solver.termination_condition)

## C.1.d. Save duals:
from pyomo.core import Constraint
obs_lu_duals=dict()
for c in fwmL.component_objects(Constraint, active=True):
    if str(c) == "c3":
        cobject = getattr(fwmL, str(c))
        for index in cobject:
            obs_lu_duals[index]=fwmL.dual[cobject[index]]

## C.1.e. 1st stage result: Calculate alpha and gamma:
#gamma1 = [((2. * a / b) if (b > 0.0) else 0.0) for a,b in zip(obs_lu_duals.values(),obs_lu.values())]
gamma1 = [((a / b) if (b > 0.0) else 0.0) for a,b in zip(obs_lu_duals.values(),obs_lu.values())]
alpha1 = [-(0.5 * a * b) for a,b in zip(gamma1,obs_lu.values())]
print(alpha1)
if run_mode == 1:
    print("+++ alpha check: +++")
    print([(a,b) for a,b in zip(alphas, alpha1)])
    print("+++ gamma check: +++")
    print([(a,b) for a,b in zip(gammas, gamma1)])
    alphas=alpha1
    linear_term_sum=[p*y - c - h*f*n - a for p,y,c,h,f,n,a in zip(prices,yields,land_costs,water_pheads,water_pcfs,water_nirs,alphas)]
    gammas=dict(enumerate(gamma1))
    net_prices=dict(enumerate(linear_term_sum))


## C.2. 2st stage: Quadratic model included in JWP model simulations 
## C.2.a. Constructing model inputs:
##  (repetition to be safe - deepcopy does not work on PYOMO models)
fwm = ConcreteModel()
fwm.ids = Set(initialize=ids)
fwm.farm_ids = Set(initialize=farm_ids)
fwm.binary_ids = Set(initialize=binary_ids)
fwm.crop_ids_by_farm = Set(fwm.farm_ids, initialize=crop_ids_by_farm)
fwm.crop_ids_by_farm_and_constraint = Set(fwm.farm_ids, fwm.binary_ids, fwm.binary_ids, initialize=crop_ids_by_farm_and_constraint)
fwm.net_prices = Param(fwm.ids, initialize=net_prices, mutable=True)
fwm.gammas = Param(fwm.ids, initialize=gammas, mutable=True)
fwm.land_constraints = Param(fwm.farm_ids, fwm.binary_ids, fwm.binary_ids, initialize=land_constraints_by_farm, mutable=True)
fwm.water_constraints = Param(fwm.farm_ids, fwm.binary_ids, fwm.binary_ids, initialize=water_constraints_by_farm, mutable=True)
fwm.nirs = Param(fwm.ids, initialize=nirs, mutable=True)
fwm.xs = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)

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

## C.2.c Creating and running the solver:
opt = SolverFactory("ipopt", solver_io='nl')
results = opt.solve(fwm, keepfiles=False, tee=True)
print(results.solver.termination_condition)


## D.1. Storing main model outputs:
result_xs = dict(fwm.xs.get_values())

## D.2. Storing derived model outputs: 
df_output=pd.DataFrame(index=range(0,(sd_no*crop_no)),columns=["subdistrict","crop","land_use","irrigation","profit",
                                                                 "alpha","gamma","land_ratio_to_obs"])
df_output_mth=pd.DataFrame(index=range(1,1+(sd_no*12*crop_no)),columns=["subdistrict","month","crop","land_use","irrigation","profit",
                                                                        "alpha","gamma"])
i=1
for sd in range(sd_no):
    for crp in range(crop_no):
        crop_id=crop_ids_by_farm[sd][crp]
        current_land_use=result_xs[crop_id] if result_xs[crop_id] else 0.0 
        current_water_use=current_land_use*gw_nirs[crop_id]
        current_profit=current_land_use*(net_prices[crop_id]+alphas[crop_id])
        current_alpha1=alpha1[crop_id]
        current_gamma1=gamma1[crop_id]
        land_ratio_to_obs = (current_land_use / obs_lu[crop_id]) if (obs_lu[crop_id] > 0.0) else int(obs_lu[crop_id] == round(current_land_use,0))
        df_output.loc[crop_id]=[subdistricts[sd],crop_types[crp],current_land_use,current_water_use,current_profit,
                                current_alpha1,current_gamma1,land_ratio_to_obs]
        for mth in range(1,13):
            water_percent=float(data_seasons["water_percent"][(data_seasons["crop"]==crop_types[crp])&(data_seasons["month"]==mth)])
            current_water_use_mth=water_percent*current_water_use
            df_output_mth.loc[i]=[subdistricts[sd],mth,crop_types[crp],current_land_use,current_water_use_mth,current_profit,
                                  current_alpha1,current_gamma1]
            i+=1
        
            
## D.3. Writing model outputs into Excel file:
out_filename = "190918-PyomoPMP-2Stages-OutputsEndog-1.xlsx" if (run_mode == 1) else "190918-PyomoPMP-2Stages-Outputs-1.xlsx"
writer = pd.ExcelWriter(out_filename)
df_output.to_excel(writer,'All_outputs_annual')
df_output_mth.to_excel(writer,'All_outputs_by_month')
writer.save()


## E. Hold until enter key is pressed:
try:
    input("Press enter to continue")
except SyntaxError:
    pass

