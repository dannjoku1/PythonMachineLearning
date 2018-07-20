import pandas
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

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

X = np.array(df.drop(['label'],1))
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
X = preprocessing.scale(X)

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#n_jobs is the amount of threads that we are willing to run at any given time
#clf = LinearRegression() will run this algorithm linearly
#clf = LinearRegression(n_jobs=10) increase jobs ran in parallel to make training faster
#clf = LinearRegression(n_jobs=-1) will run as many jobs as the processor can handle
clf = LinearRegression(n_jobs=10)
# it's good practice to train and test on different sets of data
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test )

print(accuracy)