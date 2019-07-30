#Importing Libraries
import os
import pandas as pd
import numpy as np
from fancyimpute import KNN  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#Setting work environment
os.chdir("S:/Data science/Bike Project")
os.getcwd()

#Loading data
bikedata = pd.read_csv("day.csv", sep =",")

bikedata.shape

bikedata["cnt"].describe

####################### Outlier Analysis ##################################

sns.boxplot(x=bikedata['instant'])
sns.boxplot(x=bikedata['season'])
sns.boxplot(x=bikedata['yr'])
sns.boxplot(x=bikedata['mnth'])
sns.boxplot(x=bikedata['holiday'])
sns.boxplot(x=bikedata['weekday'])
sns.boxplot(x=bikedata['workingday'])
sns.boxplot(x=bikedata['weathersit'])
sns.boxplot(x=bikedata['temp'])
sns.boxplot(x=bikedata['atemp'])
sns.boxplot(x=bikedata['hum'])
sns.boxplot(x=bikedata['windspeed'])
sns.boxplot(x=bikedata['casual'])
sns.boxplot(x=bikedata['registered'])
sns.boxplot(x=bikedata['cnt'])

#Detect and delete outliers from data
cnames = ['hum','windspeed']
for i in cnames:
     print(i)
     q75, q25 = np.percentile(bikedata.loc[:,i], [75 ,25])
     iqr = q75 - q25

     min = q25 - (iqr*1.5)
     max = q75 + (iqr*1.5)
     print(min)
     print(max)
    
     bikedata = bikedata.drop(bikedata[bikedata.loc[:,i] < min].index)
     bikedata = bikedata.drop(bikedata[bikedata.loc[:,i] > max].index)
     

######################## Missing Value Analysis #######################################
     
pd.DataFrame(bikedata.isnull().sum())

#As there is no missing value, we need to skip this process

########################### Feature Selection ########################################
##Correlation analysis
#Correlation plot
colnames = ['instant','season','yr','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','casual','registered','cnt']

df_corr = bikedata.loc[:,colnames]

#Set the width and height of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

bikedata = bikedata.drop(['instant','dteday','atemp','casual','registered'], axis=1)

########################### Feature Scaling ###########################################

#Normalization
scaler = StandardScaler()
scaler.fit(bikedata)
normscaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
  
# Scaled feature 
bikedata = normscaler.fit_transform(bikedata)

########################## Model Development ##########################################

#Using cross validation for train and test split
train, test = train_test_split(bikedata, test_size=0.4)

#Linear regression using ordinary least square method
linearmodel = sm.OLS(train[:,10], train[:,0:9]).fit()

#Summary of linear regression model
linearmodel.summary()

#Predicting through linear regression model
linearpredictions = linearmodel.predict(test[:,0:9])

#Using MAPE to find error
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape
 
MAPE(test[:,10], linearpredictions)

#MAPE = 19.22329505359607
#Accuracy = 80.7768

sns.scatterplot(x=test[:,10], y=linearpredictions)

#Decision tree for regression
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train[:,0:9], train[:,10])

#Applying model on test data
predictions_DT = fit_DT.predict(test[:,0:9])

#Using MAPE to find error
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape

MAPE(test[:,10], predictions_DT)

#MAPE = 27.453329762806504
#Accuracy = 72.5467

sns.scatterplot(x=test[:,10], y=predictions_DT)

#Random Forest for regression
randomf = RandomForestRegressor(n_estimators = 1000)

#Fitting the model
randomf.fit(train[:,0:9], train[:,10])

predictions_RF = randomf.predict(test[:,0:9])

#Using MAPE to find error
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape

MAPE(test[:,10], predictions_RF)

#MAPE = 15.435223101207582
#Accuracy = 84.5647

#Plotting the output
sns.scatterplot(x=test[:,10], y=predictions_RF)
