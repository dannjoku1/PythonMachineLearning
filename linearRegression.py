import pandas
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# pickle is the serialization of any python object such as dict or classifier

style.use('ggplot')

quandl.ApiConfig.api_key = "UhJb_S9cSjxtekqb1ops"

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) /df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) /df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
# column shifting for adjusted price changes
df['label'] = df[forecast_col].shift(-forecast_out)

# X is the features/attibutes
# y is the label/price
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] #represents 30 days of data
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# n_jobs is the amount of threads that we are willing  to run at any given time
# clf = LinearRegression() will run this algorithm linearly
# clf = LinearRegression(n_jobs=10) increase jobs ran in parallel to make training faster
# clf = LinearRegression(n_jobs=-1) will run as many jobs as the processor can handle
clf = LinearRegression(n_jobs=-1)
# it's good practice to train and test on different sets of data
clf.fit(X_train, y_train)
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test )
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

# prediction has no idea what date the prediction is for
# y is the label, otherwise known as price
# X is the the feature, and the date is not a feature
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# this function creates dates on the axis
# iterating through the forecast set, taking each forecast and day
# setting those as the values in the dataframe
# this makes the future features not not a number
# takes all first columns, sets them to nan. the final column is whatever i is
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] #df.loc[next_date] is the index

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()