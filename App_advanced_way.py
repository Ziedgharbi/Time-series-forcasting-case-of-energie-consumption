import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

%pip install -U xgboost
%pip install -U graphviz

import xgboost as xgb
from xgboost import plot_importance, plot_tree
from  sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


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



#histogramme 
data["PJME_MW"].plot(kind='hist', bins=600)

# some musures on 2011 look weird : think of outliers


data[data["PJME_MW"]<20_000].plot(style=".")

data["PJME_MW"].describe()

#drop observation below 19_000 : simple way, you could think of othrers
# like statistics test
data=data.query("PJME_MW>19_000").copy()

data[data["PJME_MW"]<20_000].plot(style=".")
data["PJME_MW"].plot(style=".").plot(style='.')


### time serie cross validation for split data, it is more robust


tscv=TimeSeriesSplit(n_splits=5,test_size=24*365*1, gap=24)

data=data.sort_index() # sort by time, essential step

# now split data

fig, ax= plt.subplots(5,1,figsize=(15,5),sharex=True)

for i, (train, test) in enumerate(tscv.split(data)):
    
    x_train=data.iloc[train]
    x_test=data.iloc[test]
    
    
    x_train["PJME_MW"].plot(ax=ax[i], 
                            label="Training set", 
                            title ="Split fold "+str(i+1))
    
    x_test["PJME_MW"].plot(ax=ax[i], 
                           label="Testing set", 
                           title ="Split fold"+str(i+1))

   
    ax[i].axvline(x_test.index.min(), color="black", ls='--')
   
plt.show()
    
  


#feature creation
data["hour"]=data.index.hour
data["dayofweek"]=data.index.dayofweek
data["dayofyear"]=data.index.dayofyear
data["month"]=data.index.month
data["quarter"]=data.index.quarter
data["year"]=data.index.year


## create lag feature 
data["lag1"]=data["PJME_MW"].shift(364) # lag 1 years
data["lag2"]=data["PJME_MW"].shift(728)  # lag 2 years
data["lag3"]=data["PJME_MW"].shift(1092) # lag 3 years



## train model for each cross validation split 







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
train["PJME_MW"].plot(ax=ax, label="Training set")
test["PJME_MW"].plot(ax=ax, label="Testing set")
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
consumption_pred=pd.DataFrame(reg.predict(X_test), index=test.index,  columns=["PJME_MW"])

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


# error calculation
error=np.sqrt(mean_squared_error(y_test, consumption_pred))


# see dates where model are worst on prediction
error =np.abs( pd.DataFrame(y_test)-consumption_pred)

error["date"]=error.index.date

error.groupby(['date']).mean().sort_values(by="PJME_MW",ascending=False).head(10)

