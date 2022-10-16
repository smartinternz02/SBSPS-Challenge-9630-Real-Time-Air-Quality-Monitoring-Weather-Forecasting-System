# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# %%
data=pd.read_csv(r'C:\Users\Admin\Downloads\Aerocharm\Dataset\city_day.csv')
data.head()


# %%
data['City'].unique()

# %%
df = data[data['City']=='Delhi']
df.head

# %%
df.info()

# %%
df.isnull().sum()

# %%
df= df.drop(['Xylene','Benzene','O3'], axis=1)

# %%
df = df.dropna()
sns.heatmap(df.corr())

# %%
df.describe()

# %%
sns.boxplot(data=df)

# %%
drop_outlier = df[(df['AQI']>500) | (df['PM2.5']>180) | (df['NO']>65) |(df['NH3']>50) | (df['NO2']>90) | (df['NOx']>100) |(df['PM10']>250)].index

# %%
df = df.drop(drop_outlier)
#DF=DF.drop(drop_outlier2)

df.info()

# %%
sns.boxplot(data= df)

# %%
"""plt.figure(figsize=(5,3),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'PM2.5', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

# %%
plt.figure(figsize=(5,3),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'NO2', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

# %%
plt.figure(figsize=(5,3),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'NH3', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

# %%
plt.figure(figsize=(5,3),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'NOx', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

# %%
plt.figure(figsize=(5,3),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'PM10', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

# %%
drop_outlier1 = df[(df['AQI']<165) & (df['PM10']>200)].index
drop_outlier2 = df[(df['AQI']>200) & (df['PM10']<110)].index
df = df.drop(drop_outlier1)
df = df.drop(drop_outlier2)

# %%
plt.figure(figsize=(5,3),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'PM10', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

# %%
plt.figure(figsize=(5,3),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'NO', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

# %%
plt.figure(figsize=(5,3),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'SO2', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

# %%
plt.figure(figsize=(5,3),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'Toluene', data=df,hue ='AQI_Bucket',palette = palette, ci= None)"""

# %%
df.info()

# %%
X= df.drop(['City', 'AQI_Bucket', 'AQI','Date'], axis= 1)

# %%
y= df['AQI']

# %%
sns.boxplot(data=X)

# %%
X.info()

# %%
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)

# %%
X.head()

# %%
print(model.feature_importances_)

# %%
feat_importances_plt = pd.Series(model.feature_importances_, index=X.columns)
feat_importances_plt.nlargest(4).plot(kind='bar')
plt.show()

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# %%
from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor()
regressor.fit(X_train,y_train)

print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))

# %%
from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=3)

# %%
score.mean()

# %%
prediction=regressor.predict(X_test)

sns.distplot(y_test-prediction)

# %%
plt.scatter(y_test,prediction)

# %%
RandomForestRegressor()
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Method of selecting samples for training each tree

# %%
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

# %%
# Create the model for tuning
rf = RandomForestRegressor()

# Random search of parameters, using 2 fold cross validation, it will search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 25, cv = 4, verbose=2, random_state=70, n_jobs = 1)

# %%
rf_random.fit(X_train,y_train)

# %%
rf_random.best_params_

# %%
rf_random.best_score_

# %%
pred_value=rf_random.predict(X_test)

# %%
sns.distplot(y_test-pred_value)

# %%
plt.scatter(y_test,prediction)

# %%
from sklearn import metrics
#Calculate RMSE( Root Mean Squared Error) for the test and predicted value
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_value)))

# %%
def calc_aqi(pm25,pm10,no,no2,nox,nh3,co,so2,tol):
    pm25 = float(pm25)
    pm10 = float(pm10)
    no = float(no)
    no2 = float(no2)
    nox = float(nox)
    nh3 = float(nh3)
    co = float(co)
    so2 = float(so2)
    tol = float(tol)
    x = [[pm25,pm10,no,no2,nox,nh3,co,so2,tol]]

    return rf_random.predict(x)

# %%
# Input sequence: pm25,pm10,no,no2,nox,nh3,co,so2,tol
pm25=(input("Enter the PM2.5 Value:"))
pm10=(input("Enter the PM10 Value:"))
no=(input("Enter the Nitro Value:"))
no2=(input("Enter the Nitrogen Di oxide Value:"))
nox=(input("Enter the Nitrogen oxide Value:"))
nh3=(input("Enter the Ammonia Value:"))
co=(input("Enter the Carbon Monoxide Value:"))
so2=float(input("Enter the Sulphur Di oxide Value:"))
tol=float(input("Enter the Toluene Value:"))
pred_value=calc_aqi(pm25,pm10,no,no2,nox,nh3,co,so2,tol)[0]
if pred_value:
  print('AQI Value is :' ,pred_value)

# %% [markdown]
# ![AQI-Chart_20.jpg](attachment:AQI-Chart_20.jpg)
filename='final_model.sav' 
joblib.dump(rf_random,filename)

# %%



