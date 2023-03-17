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
#model_path=directory_path+'model/'

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
score=[]

for i, (train, test) in enumerate(tscv.split(data)):
    
    X_train=data.iloc[train].drop("PJME_MW", axis=1)
    y_train=data.iloc[train]["PJME_MW"]
    
    X_test=data.iloc[test].drop("PJME_MW", axis=1)
    y_test=data.iloc[test]["PJME_MW"]
    
    reg=xgb.XGBRegressor(base_score=0.5,booster="gbtree",
                         n_estimators=1000, 
                         early_stopping_rounds=40,
                         objective='reg:linear',
                         max_depth=3,
                         learning_rate=0.01)
    
    reg.fit(X_train, y_train, 
            eval_set=[(X_train, y_train),(X_test,y_test)],
            verbose=True)
    
    y_pred=pd.DataFrame(reg.predict(X_test), index=y_test.index,  columns=["PJME_MW"])

    error=np.sqrt(mean_squared_error(y_test, y_pred))
    score.append(error)

print(" RSME for F fold " , np.mean(score))
print("Score for each fold ", score )



# Future prediction 

last_time=data.index.max()  #last date

future = pd.date_range(last_time,'2019-08-01',freq="H")
future_data=pd.DataFrame(index=future)

future_data["isfuture"]=True


data["isfuture"]=False


data_and_future= pd.concat([data,future_data])




#feature creation
data_and_future["hour"]=data_and_future.index.hour
data_and_future["dayofweek"]=data_and_future.index.dayofweek
data_and_future["dayofyear"]=data_and_future.index.dayofyear
data_and_future["month"]=data_and_future.index.month
data_and_future["quarter"]=data_and_future.index.quarter
data_and_future["year"]=data_and_future.index.year


## create lag feature 
data_and_future["lag1"]=data_and_future["PJME_MW"].shift(364) # lag 1 years
data_and_future["lag2"]=data_and_future["PJME_MW"].shift(728)  # lag 2 years
data_and_future["lag3"]=data_and_future["PJME_MW"].shift(1092) # lag 3 years



future_and_features= data_and_future.query('isfuture').copy()


# now prediction
future_and_features.columns
pred =pd.DataFrame( reg.predict(future_and_features.drop(["isfuture","PJME_MW"],axis=1)), index=future_and_features.index, columns=[future_and_features.columns[0]])

pred.index=future_and_features.index

pred.plot()


# save model for future use

reg.save_model("xgboost.json")
