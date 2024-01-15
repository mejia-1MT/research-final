import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Your dataset
data = {
    'product_id': [5] * 13,
    'day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'product_sold': [2000] * 13,
    'price': [850] * 13,
    'rating': [4.9] * 13,
    'total_rating': [866, 868, 868, 868, 873, 873, 874, 875, 876, 882, 882, 884, 887],
    'status': [2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2]
}
df = pd.DataFrame(data)

# Assuming 43.36 is the value you want to redistribute
value = 93.33

# Features
features = ['price', 'rating', 'total_rating', 'status']
X = df[features]

# Scale the features
scaler = MinMaxScaler()
X['scaled_price'] = 1 - (X['price'] / X['price'].max())  # Inverse scaling for price
X_scaled = scaler.fit_transform(X.drop(columns=['price']))

# Calculate weights based on the scaled features
weights = X_scaled.sum(axis=1) / X_scaled.sum(axis=1).sum()

# Check for NaN values in weights and redistribute evenly if found
if np.isnan(weights).any():
    num_days = len(weights)
    weights = np.ones(num_days) / num_days


# Train a linear regression model to predict weights
model = LinearRegression()
model.fit(X_scaled, weights)

# Predict redistribution weights
weights_predicted = model.predict(X_scaled)

# Redistribute value based on adjusted predicted weights
df['predicted_redistributed_value'] = weights_predicted * value

# Calculate proportional redistributed value for each day
df['redistributed_value'] = weights * value

# Compare the results
print(df.drop(columns=['product_id']))


# print(f'x scaled {X_scaled}')
# print(f'axis 1 {X_scaled.sum(axis=1)}')
# print(f'axis 1 sum {X_scaled.sum(axis=1).sum}')
# print(weights)
# print(df)

# # Calculate evaluation metrics
# mae = mean_absolute_error(y_test, predicted_sales)
# mse = mean_squared_error(y_test, predicted_sales)
# r2 = r2_score(y_test, predicted_sales)

# print(f"Mean Absolute Error: {mae}")
# print(f"Mean Squared Error: {mse}")
# print(f"R-squared: {r2}")