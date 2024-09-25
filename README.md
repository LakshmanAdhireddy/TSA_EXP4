## Developed By: Lakshman
## Register No: 212222240001
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 

### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
%matplotlib inline

# Read data
train = pd.read_csv('/rainfall.csv')

# Convert 'date' to datetime, let pandas infer the format automatically
train.timestamp = pd.to_datetime(train.date)
train.index = train.timestamp
train.drop("date", axis=1, inplace=True)

# Strip any leading/trailing whitespace in column names
train.columns = train.columns.str.strip()

# Plot rainfall data
train['rainfall'].plot()

# Check for stationarity using ADF test
result = adfuller(train['rainfall'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Since the p-value is > 0.05, difference the data
train_diff = train['rainfall'].diff().dropna()

# Check for stationarity again on differenced data
result = adfuller(train_diff)
print("ADF Statistic (after differencing):", result[0])
print("p-value:", result[1])

# Plot the differenced data
plt.figure(figsize=(15, 5))
plt.plot(train_diff)
plt.title('Differenced rainfall')
plt.xlabel('Date')
plt.ylabel('Rainfall')
plt.show()

# Fit ARIMA(1, 1, 1) model on the original data (auto-differencing in model)
model = ARIMA(train['rainfall'], order=(1, 1, 1))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Predict future values
start = len(train)
end = start + 2000
predictions = model_fit.predict(start=start, end=end, typ='levels')  # 'typ=levels' ensures we get original levels

# Plot the original and predicted values
plt.figure(figsize=(15, 5))
plt.plot(train['rainfall'], label='Original Series')
plt.plot(predictions, label='Predictions', color='red')
plt.legend()
plt.title('ARIMA Model Predictions on Rainfall Data')
plt.xlabel('Date')
plt.ylabel('Rainfall')
plt.show()
~~~

## OUTPUT:
## 1.Dataset:
<img width="380" alt="1" src="https://github.com/user-attachments/assets/05709ff9-8a61-4719-ac8b-3b13a294b55b">

## ADF and P values:
<img width="401" alt="2" src="https://github.com/user-attachments/assets/65188d9e-f34b-43be-85bc-8cf1754a62cb">

<img width="830" alt="3" src="https://github.com/user-attachments/assets/6d940de5-d3aa-4229-a101-242e91eb230a">

## Model summary:

<img width="476" alt="4" src="https://github.com/user-attachments/assets/3871fdba-6af5-4bb7-ba09-828cd898f9c8">

## ARMA Model:

<img width="693" alt="5" src="https://github.com/user-attachments/assets/5fb31aba-976e-42ca-aae2-34cf44849847">



## RESULT:

Thus, a python program is created to fir ARMA Model successfully.
