import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

%pip install -U xgboost
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from  sklearn.metrics import mean_squared_error

directory_path="C:/Users/pc/Nextcloud/Python/GITHUB/Time-series-forcasting-case-of-energie-consumption/"
data_path=directory_path+"data/"

# load data 
data=pd.read_csv(data_path+"PJME_hourly.csv")
data=data.set_index("Datetime")
data.index=pd.to_datetime(data.index)
data.head
data.tail
data.columns

# vizualation
data.plot(style=".", figsize=(15,5), 
          title="Energy consumption hourly")
sns.scatterplot(data.index, data.PJME_MW)

## split data 

x_train= data.loc[data.index <"01-01-2015"]
x_test= data.loc[data.index >="01-01-2015"]

#visualization train and test 

fig, ax= plt.subplots(figsize=(15,5))
x_train.plot(ax=ax, label="Training set")
x_test.plot(ax=ax, label="Testing set")
ax.axvline('01-01-2015', color="black", ls="--")
ax.legend(["Training set", " Testing set"])
plt.show()


# plot a sample to see seasonality 

data.loc[(data.index >'01-02-2010')  & (data.index <='01-10-2010')].plot()
 

#feature creation
data["hour"]=data.index.hour
data["dayofweek"]=data.index.dayofweek
data["dayofyear"]=data.index.dayofyear
data["month"]=data.index.month
data["quarter"]=data.index.quarter
data["year"]=data.index.year


#visualization of the relation between target and features

""" by hour"""
fig,ax=plt.subplots(figsize=(18,8))
sns.boxplot(data=data, x="hour", y='PJME_MW', palette="Blues")
ax.set_title("Consumption by hour")
plt.show()

""" by month"""
fig,ax=plt.subplots(figsize=(18,8))
sns.boxplot(data=data, x="month", y='PJME_MW', palette="Blues")
ax.set_title("Consumption by month")
plt.show()


""" by year"""
fig,ax=plt.subplots(figsize=(18,8))
sns.boxplot(data=data, x="year", y='PJME_MW', palette="Blues")
ax.set_title("Consumption by year")
plt.show()


### model definition 
reg=xgb.XSGBregrssor(n_estimators=10000)
reg.fit()


# plot feature importance
plot_importance(reg, hight=0.4)

