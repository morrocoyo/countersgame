import pandas as pd
import pickle
import numpy as np
import re

import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

llegada_clientes = pd.read_csv('Data/distri_pasajeros.csv')
llegada_clientes = llegada_clientes.loc[llegada_clientes['distribucion']!=0]
llegada_clientes['por_minuto']=llegada_clientes['distribucion']/15

#%%
X_train=list(range(1,30))
y_train=list(np.zeros((1,29))[0])
for r in range(len(llegada_clientes)):
    # for c in ['franja_lim_1','franja_lim_2']:
    # if c=='franja_lim_1':
    X_train.append(llegada_clientes.iloc[r]['franja_lim_1'])
    y_train.append(llegada_clientes.iloc[r]['por_minuto'])
    for k in range(1,14):
        X_train.append(llegada_clientes.iloc[r]['franja_lim_1']+k)
        y_train.append(llegada_clientes.iloc[r]['por_minuto'])
X_train=np.array(X_train)
y_train=np.array(y_train)

# Define the Gaussian Process model
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
# kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-10, 0.001))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
# Fit the model to the training data
gp.fit(X_train[:, np.newaxis], y_train)

# Generate test data
X_test = np.arange(1,316)

# Predict using the trained model
y_pred, sigma = gp.predict(X_test[:, np.newaxis], return_std=True)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='red', label='Training Data')
# plt.plot(X_test, np.array(llegada_clientes[]), c='green', label='True Function')
plt.plot(X_test, y_pred, c='blue', label='Predicted Function')
# plt.fill_between(X_test, y_pred - 2 * sigma, y_pred + 2 * sigma, color='gray', alpha=0.3, label='Uncertainty')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gaussian Process Regression')
plt.legend()
plt.show()

#%%
ypredlist = list(y_pred)
pickle.dump(ypredlist,open('gym-examples/data/ypredlist','wb'))

#%% vuelosbog = pickle.load(open('Data/Vuelos_Test.pickle','rb'))
vuelosabr=pickle.load(open('gym-examples/data/vuelos_abril.pckl','rb'))
vuelosbog=vuelosabr['0']
vuelosbog={k:v for k,v in vuelosbog.items() if v['Booked'].strip()!='0Y'}
{k:v.update({'booked':int(re.split('Y |C ',v['Booked'])[-2])}) for k,v in vuelosbog.items()}
vuelosboginter={k:v for k,v in vuelosbog.items() if v['TIPO']=='INTER'}
# sorted_vuelos = dict(sorted(vuelosbog.items(), key=lambda x: x[1]['Hora_Salida']))
# z=sorted(vuelosbog.items(), key=lambda x: x[1]['Hora'])

#%% definir funcion que calcule numero de pasajeros que llegan a fila por minuto de el dia
# yy=y_pred[::-1]
def pasajeros_fila_vuelo(v):
    dic_pas={}
    primer_min=max(1,vuelosboginter[v]['Hora']-y_pred.shape[0])
    if primer_min==1:
        yy_pred_inver=y_pred[:vuelosboginter[v]['Hora']][::-1]
    else:
        yy_pred_inver=y_pred[::-1]
    sumc=0
    h=sumc+yy_pred_inver[0]*vuelosboginter[v]['booked']
    lastcum=0
    for i in range(1,yy_pred_inver.shape[0]):
        sumc=sumc+yy_pred_inver[i]*vuelosboginter[v]['booked']
        if np.floor(sumc)>h:
            dic_pas[primer_min+i]=np.floor(sumc)-lastcum
            h=np.floor(sumc)
            lastcum+=dic_pas[primer_min+i]
    return dic_pas

dfs_queue=[pd.DataFrame(index=pasajeros_fila_vuelo(v).keys(),data=pasajeros_fila_vuelo(v).values()) for v in vuelosboginter]
sum_df = pd.concat(dfs_queue,axis=1).sum(axis=1)

sum_df.to_csv('gym-examples/data/sum_distri_dia.csv')

            
        
        



















