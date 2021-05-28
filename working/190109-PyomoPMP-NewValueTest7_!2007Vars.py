import os
import sys
global basepath
print(os.path.dirname(sys.argv[0]))
##basepath = os.path.dirname(sys.argv[0]).split(__file__)[0]
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np

## A.1. Loading main model data:
##try:
##    run_year=input("Please enter \"1\" for 2006 results, \"2\" for 2014 results:")
##except SyntaxError:
##    pass
##filename = ("181104-PyomoPMP-Inputs2006-2.xlsx" if (run_year==1)
##            else "181104-PyomoPMP-Inputs2014-3.xlsx")

#filename = "190109-PyomoPMP-NewInputs2010_1.xlsx"
#filename = "190109-PyomoPMP-NewInputs2010_1TESTFIX.xlsx"
#filename = "190109-PyomoPMP-NewInputs2010_2b_NoJVConstr.xlsx"
#filename = "190109-PyomoPMP-NewInputs2010_3b_NoJVConstr.xlsx"
# filename = "190109-PyomoPMP-NewInputs2010_7_!2007Vars.xlsx"
# print("Selected: " + filename)
# filename_and_path=os.path.join(basepath, filename)
data_file=pd.ExcelFile("190109-PyomoPMP-NewInputs2010_7_!2007Vars.xlsx")
data_profit = data_file.parse("Profit")
data_constraints = data_file.parse("Constraints")
crop_types=["Barley","OtherVegW","OtherFld","Olive","OtherTrees","OtherVegS"]
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
##water_pheads_raw=data_profit["water_phead"]
##water_pheads_offset=data_profit["water_phead_offset"]
##water_pheads=[(a+b) if ((a+b) > 0.0) else 0.0  for a,b in zip(water_pheads_raw,water_pheads_offset)]
## #water_pheads=data_profit["water_phead_2014"]
water_pheads=data_profit["water_phead"]

##profit_share=[((1.0 - ((c + h*f*n) / (p*y))) if ((p*y) > 0.0) else 1.0) for p,y,c,h,f,n in zip(prices,yields,land_costs,water_pheads,water_pcfs,water_nirs)]
##adj_factor=[(1.0 if (p >= 0.1) else (1.0 + 0.1 - p)) for p in profit_share]
##adj_prices=[(p * f) for p,f in zip(prices,adj_factor)]
##linear_term_sum=[p*y - c - h*f*n - a for p,y,c,h,f,n,a in zip(adj_prices,yields,land_costs,water_pheads,water_pcfs,water_nirs,alphas)]
linear_term_sum=[p*y - c - h*f*n - a for p,y,c,h,f,n,a in zip(prices,yields,land_costs,water_pheads,water_pcfs,water_nirs,alphas)]

## B.3. Preparing model vars and params: (these need to be dict()!)
gammas=dict(data_profit["gamma"])
nirs=dict(water_nirs)
net_prices=dict(enumerate(linear_term_sum))
x_start_values=dict(enumerate([0.0]*3))


## C.1. Constructing model inputs:
fwm = ConcreteModel()
fwm.ids = Set(initialize=ids)
fwm.farm_ids = Set(initialize=farm_ids)
fwm.binary_ids = Set(initialize=binary_ids) #JY skip for now
fwm.crop_ids_by_farm = Set(fwm.farm_ids, initialize=crop_ids_by_farm)
fwm.crop_ids_by_farm_and_constraint = Set(fwm.farm_ids, fwm.binary_ids, fwm.binary_ids, initialize=crop_ids_by_farm_and_constraint) #JY leave out binary ids
fwm.net_prices = Param(fwm.ids, initialize=net_prices, mutable=True)
fwm.gammas = Param(fwm.ids, initialize=gammas, mutable=True)
fwm.land_constraints = Param(fwm.farm_ids, fwm.binary_ids, fwm.binary_ids, initialize=land_constraints_by_farm, mutable=True)
fwm.water_constraints = Param(fwm.farm_ids, fwm.binary_ids, fwm.binary_ids, initialize=water_constraints_by_farm, mutable=True)
fwm.nirs = Param(fwm.ids, initialize=nirs, mutable=True) #JY need to figure out (links water availability constraint with irrigation requirement)
fwm.xs = Var(fwm.ids, domain=NonNegativeReals, initialize=x_start_values)

## C.2. Constructing model functions:
def obj_fun(fwm):
    return 0.00001*sum(sum((fwm.net_prices[i] * fwm.xs[i] - 0.5 * fwm.gammas[i] * fwm.xs[i] * fwm.xs[i]) for i in fwm.crop_ids_by_farm[f]) for f in fwm.farm_ids)
fwm.obj_f = Objective(rule=obj_fun, sense=maximize)

def land_constraint(fwm, ff, su, rf):
    return sum(fwm.xs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff,su,rf]) <= fwm.land_constraints[ff,su,rf]
fwm.c1 = Constraint(fwm.farm_ids, fwm.binary_ids, fwm.binary_ids, rule=land_constraint)

def water_constraint(fwm, ff, su, rf):
    return sum(fwm.xs[i]*fwm.nirs[i] for i in fwm.crop_ids_by_farm_and_constraint[ff,su,rf]) <= fwm.water_constraints[ff,su,rf]
fwm.c2 = Constraint(fwm.farm_ids, fwm.binary_ids, fwm.binary_ids, rule=water_constraint)

## C.3. Creating and running the solver:
opt = SolverFactory("ipopt", solver_io='nl')
results = opt.solve(fwm, keepfiles=False, tee=True)
print(results.solver.termination_condition)


## D.1. Storing main model outputs:
result_xs = dict(fwm.xs.get_values())
## Loading additional inputs:
perennial_land_use=data_profit["perennial_land_use"]

## D.2. Storing derived model outputs: 
df_output_mth=pd.DataFrame(index=range(1,1+(sd_no*12*crop_no)),columns=["subdistrict","month","crop","land_use","irrigation","profit",
                                                                        "perennial_land_use","perennial_irrigation","perennial_profit"])
i=1
for sd in range(sd_no):
    # land_output[subdistricts[sd]]=[result_xs[j] if result_xs[j] else 0 for j in crop_ids_by_farm[sd]]
    # water_output[subdistricts[sd]]=[(result_xs[j]*nirs[j]) if result_xs[j] else 0 for j in crop_ids_by_farm[sd]] # 
    # profit_output[subdistricts[sd]]=[(result_xs[j]*(net_prices[j]+alphas[j])) if result_xs[j] else 0 for j in crop_ids_by_farm[sd]]
    for crp in range(crop_no):
        crop_id=crop_ids_by_farm[sd][crp]
        current_land_use=result_xs[crop_id] if result_xs[crop_id] else 0.0 
        perennial_land_use=float(data_profit[(data_profit["crop"]==crop_types[crp])&(data_profit["subdistrict"]==subdistricts[sd])]["perennial_land_use"])
        current_water_use=current_land_use*gw_nirs[crop_id]
        current_profit=current_land_use*(net_prices[crop_id]+alphas[crop_id])
        current_land_use_with_perennials=(current_land_use+perennial_land_use) if ((crp == 3) or (crp == 4)) else current_land_use 
        current_water_use_with_perennials=current_land_use_with_perennials*gw_nirs[crop_id]
        current_profit_with_perennials=current_land_use_with_perennials*(net_prices[crop_id]+alphas[crop_id])
        for mth in range(1,13):
            water_percent=float(data_seasons["water_percent"][(data_seasons["crop"]==crop_types[crp])&(data_seasons["month"]==mth)])
            current_water_output=water_percent*current_water_use
            current_water_output_with_perennials=water_percent*current_water_use_with_perennials
            df_output_mth.loc[i]=[subdistricts[sd],mth,crop_types[crp],current_land_use,current_water_output,current_profit,
                                  current_land_use_with_perennials,current_water_output_with_perennials,0.0]
            i+=1
            
## D.3. Writing model outputs into Excel file:
#out_filename = ("181104-PyomoPMP-Outputs2006_4.xlsx" if (run_year==1) else "181104-PyomoPMP-Outputs2014_4.xlsx")
out_filename = "190109-PyomoPMP-Outputs2010_7_!2007Vars.xlsx"
writer = pd.ExcelWriter(out_filename)
df_output_mth.to_excel(writer,'All_outputs_by_month')
writer.save()


## E. Hold until enter key is pressed:
try:
    input("Press enter to continue")
except SyntaxError:
    pass

