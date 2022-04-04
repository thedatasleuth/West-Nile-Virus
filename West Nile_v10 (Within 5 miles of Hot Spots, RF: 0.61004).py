
# MAGIC %md West Nile Virus Project

# COMMAND ----------

# Import Packages

#Admin

import time
from datetime import datetime
%autocall 1
from geopy.distance import great_circle
from geopy.distance import vincenty

# Analysis

import pandas as pd
import numpy as np

# Modeling
from pygeohash import geohash
from sklearn.preprocessing import CategoricalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import roc_auc_score

# Plots

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider
import ipywidgets as widgets
from IPython.display import display, Markdown, Latex, Image

# COMMAND ----------

# Import Train Data Set

train = pd.read_csv('./west_nile/input/train.csv', na_values=['M'])

# COMMAND ----------

train = train.groupby(['Species','Date','Trap']).mean().reset_index()

# COMMAND ----------

train.columns

# COMMAND ----------

train.head()

# COMMAND ----------

# Import Test Data Set

test = pd.read_csv('./west_nile/input/test.csv', na_values=['M'])

# COMMAND ----------

test.head(5)

# COMMAND ----------

# Import Weather Data Set

weather = pd.read_csv('./west_nile/input/weather.csv', na_values=['M'])

# COMMAND ----------

# Do All the Things to Weather Set

weather = weather.drop(['Water1', 'Depart', 'Depth', 'CodeSum'], axis=1)

weather = weather.dropna()

weather = weather.replace('  T', 0)

station1 = weather[weather['Station']==1]

# COMMAND ----------

# Do All the Things to Train Set

# Label Encode Columns
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()

train['Species'] = encode.fit_transform(train['Species'])

train['Trap'] = encode.fit_transform(train['Trap'])

# train.drop(['Species'], axis=1, inplace=True)

# Combine Latitude and Longitude

train['LatLong'] = list(zip(train.Latitude, train.Longitude))

# Drop Unneccessary Columns

train = train.drop(['Block', 'Latitude',
                    'Longitude', 'AddressAccuracy'], axis=1)

# Merge Weather onto Train Data Set

train_weather = pd.merge(train, station1, how='left', on='Date')

# Transform Date

train_weather['Date'] = train_weather['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

train_weather['Date'] = train_weather['Date'].apply(lambda x: x.timetuple().tm_yday)
              
# Calculate Distances from 2 Major Centroids

centroid1 = (41.974689, -87.890615)
centroid2 = (41.673408, -87.599862)

distances1 = []

for i in train_weather['LatLong']:
    miles = vincenty(centroid1, i).miles
    
    distances1.append(miles)

distances2 = []

for i in train_weather['LatLong']:
    miles = vincenty(centroid2, i).miles
    
    distances2.append(miles)

# Add Those Distances to DataFrame

train_weather['Distances1'] = distances1

train_weather['Distances2'] = distances2

train_weather['Close_to_Centroid2'] = train_weather['Distances2'].map(lambda x: 1 if x < 5.0 else 0)

train_weather = train_weather.drop_duplicates()

train_weather = train_weather.fillna(0)

train_weather['WnvPresent'] = train_weather['WnvPresent'].apply(lambda x: 1 if x > 0 else 0)


# COMMAND ----------

# Do All the Things to Test Set

# Encode Species Columns

test['Species'] = encode.fit_transform(test['Species'])

test['Trap'] = encode.fit_transform(test['Trap'])

# Combine Latitude and Longitude

test['LatLong'] = list(zip(test.Latitude, test.Longitude))

# Drop Unneccessary Columns

test = test.drop(['Block', 'Latitude',
                    'Longitude', 'AddressAccuracy'], axis=1)

# Merge Weather onto Test Data Set

test_weather = pd.merge(test, station1, how='left', on='Date')

# Transform Date

test_weather['Date'] = test_weather['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

test_weather['Date'] = test_weather['Date'].apply(lambda x: x.timetuple().tm_yday)

# Calculate Distances from 2 Major Centroids

centroid1 = (41.974689, -87.890615)
centroid2 = (41.673408, -87.599862)

distances1 = []

for i in test_weather['LatLong']:
    miles = vincenty(centroid1, i).miles
    
    distances1.append(miles)

distances2 = []

for i in test_weather['LatLong']:
    miles = vincenty(centroid2, i).miles
    
    distances2.append(miles)

# Add Those Distances to DataFrame

test_weather['Distances1'] = distances1

test_weather['Distances2'] = distances2

test_weather['Close_to_Centroid2'] = test_weather['Distances2'].map(lambda x: 1 if x < 5.0 else 0)

test_weather = test_weather.drop_duplicates()

test_weather = test_weather.fillna(0)

# COMMAND ----------

train_weather.columns

# COMMAND ----------

# Drop Even More Columns

# train_weather = train_weather.drop(['NumMosquitos', 'LatLong',
#        'Station', 'Tmax', 'Tmin', 'DewPoint', 'Heat',
#        'Cool', 'Sunrise', 'Sunset', 'SnowFall', 'PrecipTotal', 'StnPressure',
#        'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed'], axis=1)

# COMMAND ----------

train_weather.head()

# COMMAND ----------

train_weather.columns

# COMMAND ----------

# Feature Selection

# define target
target = 'WnvPresent'

# instantiate selector
selector = SelectKBest(score_func=f_classif, k=10)

# subset training data without 'drops'
# train_features = train_weather.drop(drops, axis=1)
train_features = train_weather.drop('WnvPresent', axis=1).select_dtypes(include=['number'])

# subset training target
train_target = train_weather[target]

# fit selector
selector.fit(train_features, train_target)

# extract best feature indexes
best_features = selector.get_support(indices=True)

# convert indexes to feature names
features = list(train_features.columns[selector.get_support(indices = True)])
print(features)

# COMMAND ----------

train_weather.columns

# COMMAND ----------

# Train-Train-Split on Data Set

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

X = train_weather[['Species', 'Date', 'Trap',
       'Station', 'Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'Heat',
       'Cool', 'Sunrise', 'Sunset', 'SnowFall', 'PrecipTotal', 'StnPressure',
       'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed', 'Distances1',
       'Distances2', 'Close_to_Centroid2']]
y = train_weather['WnvPresent']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Standard Scaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# COMMAND ----------

# Try to Balance Classes with SMOTEENN

from sklearn.metrics import recall_score
from imblearn.combine import SMOTEENN

sm = SMOTEENN()

X_train, y_train = sm.fit_sample(X_train, y_train)

# COMMAND ----------

# RandomForestClassifier

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline

rf = RandomForestClassifier()

rf_pipe = Pipeline([
    ('ss', ss),
    ('rf', rf)
])

params = {'rf__n_estimators' : [10, 15, 20],
          'rf__max_depth' : [None, 2, 3, 4, 5]}

rf_gs = GridSearchCV(rf_pipe, param_grid=params, cv=5, scoring='roc_auc')
rf_gs.fit(X_train, y_train)

best_rf_gs = rf_gs.best_estimator_

rf_gs_train = best_rf_gs.score(X_train, y_train)
rf_gs_test = best_rf_gs.score(X_test, y_test)

# COMMAND ----------

# XGBoost Classifier

gs_params = {
    'max_depth':[1, 2, 3, 4, 5],
    'n_estimators':range(1, 10, 1),
    'learning_rate':np.logspace(-5,0,5),
    'silent' : [False],
    'booster' : ['gbtree', 'gblinear', 'dart'] 
}

xgb_gs = GridSearchCV(XGBClassifier(), gs_params, cv=5, verbose=1, scoring='roc_auc')

xgb_gs = xgb_gs.fit(X_train, y_train)

best_xgb_gs = xgb_gs.best_estimator_

xgb_gs_train = best_xgb_gs.score(X_train, y_train)
xgb_gs_test = best_xgb_gs.score(X_test, y_test)

# COMMAND ----------

# BalancedBaggingClassifier

gs_params = {
    'n_estimators' : range(1, 10, 1),
#     'max_samples' : range(1, 10),
#     'max_features' : range(1, 10),
    'warm_start' : [True, False] 
}

bbc_gs = GridSearchCV(BalancedBaggingClassifier(), gs_params, scoring='roc_auc', 
                      cv=5, verbose=1)

bbc_gs = bbc_gs.fit(X_train, y_train)

best_bbc_gs = bbc_gs.best_estimator_

bbc_train = best_bbc_gs.score(X_train, y_train)
bbc_test = best_bbc_gs.score(X_test, y_test)

# COMMAND ----------

X_train.shape

# COMMAND ----------

# Executive Summary of Models

print('GridSearchCV across Random Forest:')
print(f"Best Parameters = {rf_gs.best_params_}")
print(f"Best CV Score = {rf_gs.best_score_}")
print(f"Train Score = {rf_gs_train}")
print(f"Test Score = {rf_gs_test}")
print()
print('GridSearchCV across XGBoost:')
print(f"Best Parameters = {xgb_gs.best_params_}")
print(f"Best CV Score = {xgb_gs.best_score_}")
print(f"Train Score = {xgb_gs_train}")
print(f"Test Score = {xgb_gs_test}")
print()
print('GridSearchCV across BalancedBaggingClassifier:')
print(f"Best Parameters = {bbc_gs.best_params_}")
print(f"Best CV Score = {bbc_gs.best_score_}")
print(f"Train Score = {bbc_train}")
print(f"Test Score = {bbc_test}")

# COMMAND ----------

def feat_equalize(train, test):
    Tr = X
    Te = test_weather.columns

    # remove any columns in Te that are not in Tr
    Te_not_Tr = [c for c in Te if c not in Tr]
    test_cut = test.drop(Te_not_Tr, axis=1)

    # create column of zeroes in test for any columns in Tr not in Te
    Tr_not_Te = [c for c in Tr if c not in Te]
    for c in Tr_not_Te:
        test_cut[c] = 0

    return test_cut

# COMMAND ----------

train_weather.head()

# COMMAND ----------

TEST_CUT = feat_equalize(train_weather, test_weather)
TEST_CUT.head()

# COMMAND ----------

TEST_CUT = TEST_CUT[train_weather.drop(['WnvPresent', 'NumMosquitos', 'LatLong'], axis=1).columns]

# COMMAND ----------

print(train_weather.columns.value_counts().sum())
print(train_weather.shape)
print(train_weather.columns)

# COMMAND ----------

print(TEST_CUT.columns.value_counts().sum())
print(TEST_CUT.shape)
print(TEST_CUT.columns)

# COMMAND ----------

# Predict on Test Data using Balanced Bagging Classifer model

yhat_bbc = pd.DataFrame(best_bbc_gs.predict_proba(TEST_CUT.values))
print(yhat_bbc.describe())
print()
yhat_bbc['WnvPresent'] = yhat_bbc[1].map(lambda x: 1 if x > yhat_bbc[1].mean() else 0)
                                         
print(yhat_bbc.sum())

yhat_bbc = yhat_bbc.drop([0, 1], axis=1)

# COMMAND ----------

# Predict on Test Data using XGBoost model

yhat_xgb = pd.DataFrame(best_xgb_gs.predict_proba(TEST_CUT.values))
print(yhat_xgb.describe())
print()
yhat_xgb['WnvPresent'] = yhat_xgb[1].map(lambda x: 1 if x > yhat_xgb[1].mean() else 0)
print(yhat_xgb.sum())

yhat_xgb = yhat_xgb.drop([0, 1], axis=1)

# COMMAND ----------

# Predict on Test Data using Random Forest model

yhat_rf = pd.DataFrame(best_rf_gs.predict_proba(TEST_CUT.values))
print(yhat_rf.describe())
print()
yhat_rf['WnvPresent'] = yhat_rf[1].map(lambda x: 1 if x > yhat_rf[1].mean() else 0)
print(yhat_rf.sum())

yhat_rf = yhat_rf.drop([0, 1], axis=1)

# COMMAND ----------

# converts prediction output to appropriate kaggle format

def kagglizer(pred):
    pred_format = pd.DataFrame(pred).reset_index()
    pred_format['index'] = pred_format['index']+1
    pred_format = pred_format.rename(columns={'index':'Id',0:'WnvPresent'}).set_index('Id')
    return pred_format

# COMMAND ----------

# Submission for XGBoost Model - 0.54499

submission_yhat_xgb = kagglizer(yhat_xgb)

submission_yhat_xgb.to_csv('./submission_yhat_xgb')

# COMMAND ----------

# Submission for Balanced Bagging Model - 0.49564

submission_yhat_bbc = kagglizer(yhat_bbc)

submission_yhat_bbc.to_csv('./submission_yhat_bbc')

# COMMAND ----------

# Submission for Random Forest Model - 0.61004

submission_yhat_rf = kagglizer(yhat_rf)

submission_yhat_rf.to_csv('./submission_yhat_rf')

# COMMAND ----------

def mtrx(model, X, y):
    print('score:')
    print(model.score(X,y))
    print('recall:')
    print(recall_score(y,model.predict(X)))
    print('AUC:')
    print(roc_auc_score(y, model.predict(X)))
    return

# COMMAND ----------

mtrx(best_rf_gs, X_test, y_test)

# COMMAND ----------

mtrx(best_xgb_gs, X_test, y_test)

# COMMAND ----------

mtrx(best_bbc_gs, X_test, y_test)

# COMMAND ----------


