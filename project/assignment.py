import kmean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds = pd.read_csv('ph-data.csv')


X = ds.iloc[:, :-1].values
y = ds.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
#X = sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

SEED = 1

#variable initiation.
rf = RandomForestRegressor(random_state = SEED)
dt = DecisionTreeRegressor(random_state = 0)
svr = SVR(kernel = 'rbf')
ridge = Ridge(alpha=0.1, normalize=True)
lr = LinearRegression()

#storing in an array.
regressors = [#('Multiple Linear Regression', lr),
              ('SVM', svr),
              ('Decision Tree', dt),
              ('Random Forest', rf)]


input= kmean.color
input=input.reshape(1, -1)
#input = sc_X.fit_transform(input)


sum=0

d=0
norm=0

for reg_name, reg in regressors:
    
    #fit the model.
    reg.fit(X_train, y_train)
    
    #predicting test set results.
    y_pred = reg.predict(X_test)

    # print(y_pred)
    # print(y_test)
    # print(X_test)
    
    #reshaping into former form.
    np.set_printoptions(precision=5)
    np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)    
    norm= norm+r2_score(y_test, y_pred)
    


for reg_name, reg in regressors:
    
    #fit the model.
    reg.fit(X_train, y_train)
    
    #predicting test set results.
    y_pred = reg.predict(X_test)

    # print(y_pred)
    # print(y_test)
    # print(X_test)
    
    #reshaping into former form.
    np.set_printoptions(precision=5)
    np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

    print("model name:")
    print('{:s} : {:f}'.format(reg_name, r2_score(y_test, y_pred)))
    d= d+r2_score(y_test, y_pred)
    ph= reg.predict(input)

    print("\nph is:")
    print(ph)
    print("\n")

    val=ph[0]
    
    sum=sum + r2_score(y_test, y_pred) /norm* val







# import joblib
# joblib.dump(lr, 'model.pkl')
# ['model.pkl']



# model = joblib.load('model.pkl')
#pred = model.predict(X)[-1]

print('\nour prediction:\n')
print(sum)

#print(pred)





