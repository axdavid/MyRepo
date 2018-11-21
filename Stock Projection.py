import quandl
import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm

dfStockData = quandl.get("WIKI/AMZN")

dfPastEodPrice = dfStockData[['Adj. Close']]

forecast_out = int(30)

dfPastEodPrice['Prediction'] = dfPastEodPrice[['Adj. Close']].shift(-forecast_out)

X = np.array(dfPastEodPrice.drop(['Prediction'],1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(dfPastEodPrice['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)

confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)