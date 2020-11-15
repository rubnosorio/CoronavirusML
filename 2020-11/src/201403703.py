import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

lrm = linear_model.LinearRegression()

data = pd.read_csv('ITALY-QUARANTINE.csv')
df = pd.DataFrame(data)

x = np.array(df['Quarantine_days'])
y = df['New_cases']

X=x[:,np.newaxis]



print(x)

while True:
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    mlr=MLPRegressor(solver="lbfgs",alpha=0.1,hidden_layer_sizes=(3,3), random_state=1, max_iter=2000)
    mlr.fit(X_train, y_train)
    print(mlr.score(X_train, y_train))
    if mlr.score(X_train, y_train) > 0.75:
        break

print('NUMERO DE CASOS PARA MAS DIAS DE CUARENTENA: ')
print(mlr.predict(np.array(150).reshape(1, 1))) #en 100 dias de cuartentena 