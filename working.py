import pandas as pd

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price': [22000,25000,27000,35000]
        }

df = pd.DataFrame(Cars,columns= ['Brand', 'Price'])

df.to_csv('pandas_test.csv')

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

with open('net_prices.p', 'wb') as fp:
    pickle.dump(net_prices, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('gammas.p', 'rb') as fp:
    test = pickle.load(fp)


for i in land_constraints_by_farm:
    land_constraints_by_farm[i] = land_constraints_by_farm[i][i]

water_constraints_by_farm = dict.fromkeys(water_constraints_by_farm, 9999999999)
