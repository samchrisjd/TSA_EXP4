# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 07.10.2025



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
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('/content/World_Population.csv')

numeric_cols = data.select_dtypes(include=[np.number]).columns
X = data[numeric_cols[0]].dropna()   
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]

plt.plot(X)
plt.title('Original Data')
plt.show()

max_lags = min(len(X)//2, 10)

plt.subplot(2, 1, 1)
plot_acf(X, lags=max_lags, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=max_lags, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params.get('ar.L1', 0)
theta1_arma11 = arma11_model.params.get('ma.L1', 0)

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()

plot_acf(ARMA_1, lags=20)
plt.show()

plot_pacf(ARMA_1, lags=20)
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params.get('ar.L1', 0)
phi2_arma22 = arma22_model.params.get('ar.L2', 0)
theta1_arma22 = arma22_model.params.get('ma.L1', 0)
theta2_arma22 = arma22_model.params.get('ma.L2', 0)

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*5)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()

plot_acf(ARMA_2, lags=20)
plt.show()

plot_pacf(ARMA_2, lags=20)
plt.show()
```

## OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

<img width="1235" height="651" alt="image" src="https://github.com/user-attachments/assets/b7b1f934-516b-46a0-8b0c-b4e3eca5c6b2" />


Partial Autocorrelation

<img width="1284" height="660" alt="image" src="https://github.com/user-attachments/assets/6fe34aa8-8543-463e-89a6-8602f75a9a1d" />


Autocorrelation

<img width="1307" height="667" alt="image" src="https://github.com/user-attachments/assets/3664023b-8936-4eb5-81c2-7e16c84570f0" />


SIMULATED ARMA(2,2) PROCESS:

<img width="1304" height="663" alt="image" src="https://github.com/user-attachments/assets/deb08281-dff0-49ab-a751-8c47c6350c25" />


Partial Autocorrelation

<img width="1284" height="660" alt="image" src="https://github.com/user-attachments/assets/0360f50c-c31a-4b18-9711-04d4d0e87fe7" />


Autocorrelation

<img width="1307" height="667" alt="image" src="https://github.com/user-attachments/assets/efd4fae7-ef59-4c04-b070-0a2e45ec6a0b" />

## RESULT:

Thus a python program is created to fir ARMA Model successfully.
