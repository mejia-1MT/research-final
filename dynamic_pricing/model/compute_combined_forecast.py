import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d 
import warnings

# Function to forecast demand using ARIMA for a specific product ID
def forecast_arima(train_data, product_id):
    product_data = train_data[train_data['product_id'] == product_id].copy()
    product_data['day'] = pd.to_datetime(product_data['day'])
    product_data.set_index('day', inplace=True)
    time_series = product_data['product_sold']

    # Ensure the index is a DateTimeIndex
    time_series.index = pd.date_range(start=time_series.index[0], periods=len(time_series))

    model = ARIMA(time_series, order=(5, 1, 0))

    # Suppress the convergence warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arima_model = model.fit()

    # Forecast using the proper DateTime index for the next 30 days
    forecast_index = pd.date_range(start=time_series.index[-1], periods=30)
    forecast = arima_model.forecast(steps=30, index=forecast_index)

    return forecast

# Function to predict demand using RandomForestRegressor
def predict_demand_last_30_days(train_data, product_id, features, target):
    product_data = train_data[train_data['product_id'] == product_id].copy()
    warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names*")
    # Separating features and target for the specified product
    X_train = product_data[features]
    y_train = product_data[target]
    # print("Columns in X_train:", X_train.columns)
    # print("Columns in test dataset:", product_data[features].columns)
    
    # Use the last 30 days' features as the basis for prediction
    last_features = X_train.iloc[-30:].values
    # Train the RandomForestRegressor model
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)  # Pass feature names here
    
    # Predict demand for the last 30 days within historical data
    predictions = []
    for features_30th_day in last_features:
        # Pass feature names to the predict method
        next_day_demand = rf_model.predict([features_30th_day])
        
        predictions.append(next_day_demand[0])  # Append the prediction to the list
    
    # Calculate MSE
    mse = mean_squared_error(y_train[-30:], predictions)

    # print("Time period for MSE calculation:")
    # print(y_train.index[-30:], "\n")
    # print("Actual time frame in data:")
    # print(product_data.index[-30:])

    
    
    return predictions, mse

def demand_forecast(data):
    combined_forecasts = []
    mse_values = []

    ml_features = ['price', 'rating', 'total_rating', 'status', 'customer']  # Update with relevant columns
    ml_target = 'product_sold'

    for product_id in data['product_id'].unique():
        product_data = data[data['product_id'] == product_id].copy()

        # ARIMA forecast
        arima_forecast = forecast_arima(product_data, product_id)

        # RandomForestRegressor forecast
        ml_predictions, mse = predict_demand_last_30_days(product_data, product_id, ml_features, ml_target)

        # Interpolating ARIMA forecast to match the length of ML predictions
        arima_interpolated = interp1d(np.linspace(0, 1, len(arima_forecast)), arima_forecast)
        arima_forecast_resized = arima_interpolated(np.linspace(0, 1, len(ml_predictions)))

        # Combine forecasts using a 70:30 ratio
        ml_predictions = np.array(ml_predictions)
        combined_forecast = 0.7 * arima_forecast_resized + 0.3 * ml_predictions

        # Append combined forecast and MSE values to lists for each product
        combined_forecasts.extend(combined_forecast.tolist())
        mse_values.extend([mse] * len(combined_forecast))

    # Add combined forecast and MSE values as new columns to the DataFrame
    data['demand'] = combined_forecasts
    #data['mse'] = mse_values


    # print("ARIMA Forecast:", arima_forecast)
    # print("ML Predictions:", ml_predictions)
    # print("ARIMA Resized:", arima_forecast_resized)


    return data