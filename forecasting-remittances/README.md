Time Series Forecasting of Remittances to Mexico

A full end-to-end time series analysis and forecasting project using ARIMA and SARIMA models.
This project was originally completed as part of my Post-Master programme in Data Science & Business Analytics.

Objective

Forecast future remittances sent to Mexico using 27 years of monthly data from Banco de México (1995–2022).
This problem is economically relevant because remittances are a major source of household income in Mexico and the wider CADPR region.

Methods

Exploratory time series analysis

ADF stationarity testing

Trend & seasonal decomposition

Log transformation

Differencing (regular + seasonal)

ACF & PACF diagnostics

ARIMA & SARIMA model comparison

Selection via AIC, BIC, and error metrics

Forecasting 48 months ahead

Tools

R

forecast package

stats package

ggplot2

Time-series diagnostics

Key Results

Best model selected: SARIMA(1,1,2)(0,1,1)12

Strong short-term forecasting accuracy on test set

Robust methodology applied to a real economic dataset

Final model predicts remittance growth through 2026 with confidence intervals
