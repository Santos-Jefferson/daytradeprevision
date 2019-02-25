import quandl, math, datetime
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import BDay, CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import sys

# CONSTANTS
PERIODS = 36
CROSS_TEST = 0.2


# Source data
# filename = 'novo_datasource_teste.csv'
filename = 'indfut_15min.txt'

# Importing and normalizing the .CSV file
df = pd.read_csv(filename, delimiter='\t', parse_dates=True, dayfirst=True, decimal=',', index_col=0,
                 usecols=['Data','Abertura','Máxima','Fechamento','VWAP D','Pivot']
                 )
# df.index = pd.to_datetime(df.index)
df = df.sort_index()
df = df.replace(0, np.nan)  # Replacing '0 values' to Numpy nan
df = df.dropna()  # Dropping nan rows from the dataframe
# df['Data'] = pd.to_datetime(df['Data'])  # Converting 'Data' column to datetime type
# df = df.sort_values(by=['Data'])  # Sorting Dataframe by 'Data' (oldest to newest)

# Printing for verification/audit
print(df.dtypes)
print(df.head().to_string())
print(df.tail().to_string())
# print(df['Data'].dt.hour)
# sys.exit()

# Defining min, max (from .CSV file) and today dates
# min_date_datasource = df['Data'].min()
# max_date_datasource = df['Data'].max()
today_datetime = dt.datetime.now()

# dataframe original with all prices and dates
df_preco_fech_real = df[['Fechamento']]  # Two brackets create pd Dataframe, one bracket creates pd Series.

# Creating two features to add to our dataframe
# Percentage volatility of the prices and percentage change of the prices
df['pct_vol'] = (df['Máxima'] - df['Fechamento']) / df['Fechamento'] * 100.0
df['pct_change'] = (df['Fechamento'] - df['Abertura']) / df['Abertura'] * 100.0

# Creating three dataframes for the three classifiers to be used
# df_lin = df[
#     ['Abertura', 'Maxima', 'Minima', 'Fechamento', 'VWAPD', 'MediaMovelE9', 'MediaMovelA200', 'PriorCote',
#      'MediaMovelA42', 'Lowest20', 'BBAUp20', 'BBDown20', 'Highest20', 'QAAD', 'IFR4', 'VolumeFinanceiro',
#      'MediaMovelVolA20', 'pct_change', 'pct_vol']]
# df_svr_lin = df[
#     ['Abertura', 'Maxima', 'Minima', 'Fechamento', 'VWAPD', 'MediaMovelE9', 'MediaMovelA200', 'PriorCote',
#      'MediaMovelA42', 'Lowest20', 'BBAUp20', 'BBDown20', 'Highest20', 'QAAD', 'IFR4', 'VolumeFinanceiro',
#      'MediaMovelVolA20', 'pct_change', 'pct_vol']]
# df_svr_rbf = df[
#     ['Abertura', 'Maxima', 'Minima', 'Fechamento', 'VWAPD', 'MediaMovelE9', 'MediaMovelA200', 'PriorCote',
#      'MediaMovelA42', 'Lowest20', 'BBAUp20', 'BBDown20', 'Highest20', 'QAAD', 'IFR4', 'VolumeFinanceiro',
#      'MediaMovelVolA20', 'pct_change', 'pct_vol']]

df_lin = df.copy()
df_svr_lin = df.copy()
df_svr_rbf = df.copy()

# Defining the forecast column, in this case the Adjacent Closed Price
forecast_col = 'Fechamento'

# # Define the periods of 15 minutes to forecast (36 is equal 1 day)
periods_forecast = PERIODS
forecast_out = int(periods_forecast)
#
# Shifiting the new column "label" accordingly the days forecasted above
df_lin['label'] = df_lin[forecast_col][:-forecast_out]
df_svr_lin['label'] = df_svr_lin[forecast_col][:-forecast_out]
df_svr_rbf['label'] = df_svr_rbf[forecast_col][:-forecast_out]

print(df_lin.head().to_string())
print(df_lin.tail(40).to_string())
# sys.exit()

# Creating the X dataset without the 'label' and 'Data' columns
X_lin = np.array(df_lin.drop(['label'], 1))
X_svr_lin = np.array(df_svr_lin.drop(['label'], 1))
X_svr_rbf = np.array(df_svr_rbf.drop(['label'], 1))

print(X_lin)

# Scaling the data
X_lin = preprocessing.scale(X_lin)
X_svr_lin = preprocessing.scale(X_svr_lin)
X_svr_rbf = preprocessing.scale(X_svr_rbf)
print()
print("preprocessing")
print(X_lin)

# Removing from the bottom 'forecast_out' N periods set up above from Numpy array
# Here we are preparing the X arrays to be used on cross validation, with no 'forecast_out' that was set
X_lin = X_lin[:-forecast_out:]
X_svr_lin = X_svr_lin[:-forecast_out:]
X_svr_rbf = X_svr_rbf[:-forecast_out:]
print()
print(":-forecast_out:")
print(X_lin)

# Creating the X to put the predicted data
# Getting the last rows N forecast_out from the bottom
X_lately_lin = X_lin[-forecast_out:]
X_lately_svr_lin = X_svr_lin[-forecast_out:]
X_lately_svr_rbf = X_svr_rbf[-forecast_out:]
print("X_lin")
print(X_lin)

print("X_lately_lin")
print(X_lately_lin)

# Preparing the data to drop NAs to cross validation
df_lin.dropna(inplace=True)
df_svr_lin.dropna(inplace=True)
df_svr_rbf.dropna(inplace=True)

# Creating the target array from the 'label' column
# This array has the real/correct 'fechamento' price with no 'forecast_out' that was set before
y_lin = np.array(df_lin['label'])
y_svr_lin = np.array(df_svr_lin['label'])
y_svr_rbf = np.array(df_svr_rbf['label'])
# print(y_lin)


# Cross validation
print(X_lin)
print(y_lin)

X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(X_lin, y_lin, test_size=CROSS_TEST)
X_svr_lin_train, X_svr_lin_test, y_svr_lin_train, y_svr_lin_test = train_test_split(X_svr_lin, y_svr_lin, test_size=CROSS_TEST)
X_svr_rbf_train, X_svr_rbf_test, y_svr_rbf_train, y_svr_rbf_test = train_test_split(X_svr_rbf, y_svr_rbf, test_size=CROSS_TEST)

# The classifiers
clf_lin = LinearRegression()
clf_svr_lin = svm.SVR(kernel='linear', C=1e3, verbose=False)
clf_svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.00015, verbose=False)

# Fitting the training data
clf_lin.fit(X_lin_train, y_lin_train)
clf_svr_lin.fit(X_svr_lin_train, y_svr_lin_train)
clf_svr_rbf.fit(X_svr_rbf_train, y_svr_rbf_train)

# Getting the accuracies
clf_lin.score(X_lin_test, y_lin_test)
clf_svr_lin.score(X_svr_lin_test, y_svr_lin_test)
clf_svr_rbf.score(X_svr_rbf_test, y_svr_rbf_test)
accuracy_lin = clf_lin.score(X_lin_test, y_lin_test)
accuracy_svr_lin = clf_svr_lin.score(X_svr_lin_test, y_svr_lin_test)
accuracy_svr_rbf = clf_svr_rbf.score(X_svr_rbf_test, y_svr_rbf_test)

# Predicting the data
forecast_predicted_lin = clf_lin.predict(X_lately_lin)
forecast_predicted_svr_lin = clf_svr_lin.predict(X_lately_svr_lin)
forecast_predicted_svr_rbf = clf_svr_rbf.predict(X_lately_svr_rbf)

print(X_lately_lin)
print(forecast_predicted_svr_lin)
print(forecast_predicted_svr_rbf)

# Creating the column with nan
df_lin['Forecast_lin'] = np.nan
df_svr_lin['Forecast_svr_lin'] = np.nan
df_svr_rbf['Forecast_svr_rbf'] = np.nan

print(df_lin.head().to_string())
print(df_lin.tail().to_string())

# Getting the business and holiday days
bday_us = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# Last date on dataset in accordingly with forecasted days inserted
last_date_lin = df_lin.iloc[-1].name
last_date_svr_lin = df_svr_lin.iloc[-1].name
last_date_svr_rbf = df_svr_rbf.iloc[-1].name
next_period = dt.timedelta(minutes=15)
next_day_period = last_date_lin + next_period
next_day_period_svr_lin = last_date_svr_lin + next_period
next_day_period_svr_rbf = last_date_svr_rbf + next_period

# Looping to adding every predicted price in the right date
for i in forecast_predicted_lin:
    next_date_lin = next_day_period
    next_day_period += next_period
    df_lin.loc[next_date_lin] = [np.nan for _ in range(len(df_lin.columns) - 1)] + [i]
for i in forecast_predicted_svr_lin:
    next_date_svr_lin = next_day_period_svr_lin
    next_day_period_svr_lin += next_period
    df_svr_lin.loc[next_date_svr_lin] = [np.nan for _ in range(len(df_svr_lin.columns) - 1)] + [i]
for i in forecast_predicted_svr_rbf:
    next_date_svr_rbf = next_day_period_svr_rbf
    next_day_period_svr_rbf += next_period
    df_svr_rbf.loc[next_date_svr_rbf] = [np.nan for _ in range(len(df_svr_rbf.columns) - 1)] + [i]

# Console log to accuracies
print()
print("Accuracy     Lin: ", forecast_out, " Periods ahead: ", (accuracy_lin * 100).round(2), "%")
print("Accuracy SVR lin: ", forecast_out, " Periods ahead: ", (accuracy_svr_lin * 100).round(2), "%")
print("Accuracy SVR rbf: ", forecast_out, " Periods ahead: ", (accuracy_svr_rbf * 100).round(2), "%")

print(df_preco_fech_real['Fechamento'])
print(df_lin['Forecast_lin'].tail(37))
print(df_lin.tail(37).to_string())
# print(df_svr_lin['Forecast_svr_lin'])
# print(df_svr_rbf['Forecast_svr_rbf'])

# Ploting the data to compare the predicted with the real data
df_preco_fech_real['Fechamento'].plot(color="green", linewidth=3)
df_lin['Forecast_lin'].plot(color="red")
df_svr_lin['Forecast_svr_lin'].plot(color="blue")
df_svr_rbf['Forecast_svr_rbf'].plot(color="orange")
# plt.ylim(df_lin['Forecast_lin'].min(), df_lin['Forecast_lin'].max())
# plt.xlim(left=df_lin.index., right=df_lin.index.max())
plt.legend(loc=4)
plt.title(str(forecast_out) + " Periods(s) forecast\n")
plt.suptitle("Accuracy Lin:     " + str((accuracy_lin * 100).round(2)) + "%\n" +
             "Accuracy SVR Lin: " + str((accuracy_svr_lin * 100).round(2)) + "%\n" +
             "Accuracy SVR Rbf: " + str((accuracy_svr_rbf * 100).round(2)) + "%\n\n"
             )
plt.xlabel('\nDate and Time')
plt.ylabel('Price in U$')
plt.show()
