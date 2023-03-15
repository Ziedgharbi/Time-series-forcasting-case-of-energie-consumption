import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

%pip install -U xgboost
%pip install -U graphviz

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


## split data 

train= data.loc[data.index <"01-01-2015"]
test= data.loc[data.index >="01-01-2015"]

#visualization train and test 

fig, ax= plt.subplots(figsize=(15,5))
train.plot(ax=ax, label="Training set")
test.plot(ax=ax, label="Testing set")
ax.axvline('01-01-2015', color="black", ls="--")
ax.legend(["Training set", " Testing set"])
plt.show()


# plot a sample to see seasonality 
data.columns
data.loc[(data.index >'01-02-2010')  & (data.index <='01-10-2010')]["PJME_MW"].plot()
 

### model definition 
X_train=train.drop(["PJME_MW",], axis=1)
y_train=train["PJME_MW"]

X_test=test.drop(["PJME_MW",], axis=1)
y_test=test["PJME_MW"]


reg=xgb.XGBRegressor(n_estimators=10000, early_stopping_rounds=40,
                     eta=0.01)
reg.fit(X_train, y_train, 
        eval_set=[(X_train, y_train),(X_test,y_test)],
        verbose=True)


# plot feature importance
importance=pd.DataFrame(data=reg.feature_importances_,index=reg.feature_names_in_,columns=["Score F"])

importance.sort_values('Score F').plot(kind='barh', title="feature importance")

plot_importance(reg, importance_type='gain')



# prediction and visualization all
consumption_pred=pd.DataFrame(reg.predict(x_test), index=test.index)

ax=data['PJME_MW'].plot()  
consumption_pred.plot(ax=ax)      
plt.show()                     

#ou bien
plt.plot(data['PJME_MW'])
plt.plot(consumption_pred, )
plt.show()


#visualization part of prevision
ax=data[(data.index>"02-01-2018") & (data.index <= "02-08-2018")]['PJME_MW'].plot()
consumption_pred[(consumption_pred.index>"02-01-2018") & (consumption_pred.index <= "02-08-2018")].plot(ax=ax, style='.')  
plt.legend(["Real", 'Pred'])    
plt.show() 




