import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

def calculate_customer_value(product_df, num_days=30):
    customer_value = []  # Initialize an empty list to store customer values
    
    # Extract the 'product_sold' column values from the provided DataFrame and convert to a list
    product_sold_values = product_df['product_sold'].tolist()

    # Rule 1: Check if product_sold remained constant throughout the given number of days
    if len(set(product_sold_values)) == 1:
        if product_sold_values[0] > 1000:
            # Set customer value to 50 for the changing value
            customer_value.extend([0] * (num_days - 1))
            customer_value.append(50)
        else:
            # If product_sold is not higher than 1000, set customer value to 0 for all days
            customer_value.extend([0] * num_days)
    else:
        # If product_sold changed, set customer value to 0 for all days
        customer_value.extend([0] * num_days)
        
        # Rule 2: Calculate customer value for the first change if product_sold for that day is higher than 1000
        first_change_index = next((i for i, v in enumerate(product_sold_values) if v != product_sold_values[0] and v > 1000), None)
        if first_change_index is not None and first_change_index <= num_days:
            customer_value[first_change_index - 1] = (first_change_index / num_days) * 100

            # Find subsequent changes and calculate customer value accordingly until reaching the given number of days or another change
            current_index = first_change_index
            while current_index < num_days:
                next_change_index = next((i for i in range(current_index + 1, num_days) if product_sold_values[i] != product_sold_values[current_index] and product_sold_values[i] > 1000), None)
                if next_change_index is not None:
                    if current_index == first_change_index and next_change_index == num_days - 1:
                        customer_value[next_change_index - 1] = (next_change_index / num_days) * 100
                    else:
                        customer_value[next_change_index - 1] = 100
                    current_index = next_change_index
                else:
                    break
            
            # Check for changes between day 1 and the given number of days
            if product_sold_values[0] != product_sold_values[num_days - 1] and product_sold_values[num_days - 1] > 1000:
                days_taken = num_days - current_index
                customer_value[num_days - 1] = (days_taken / num_days) * 100

        # Rule 3: Calculate customer value if product_sold is less than 1000
        for i in range(1, num_days):
            if product_sold_values[i] <= 1000:
                customer_value[i] = product_sold_values[i] - product_sold_values[i - 1]
    return customer_value

def redistribute_customer_value(df):
    unique_combinations = df[df['product_sold'] > 1000].groupby(['product_id', 'product_sold']).size().reset_index().drop(0, axis=1)

    for index, row in unique_combinations.iterrows():
        product_id = row['product_id']
        product_sold = row['product_sold']

        product_subset = df[(df['product_id'] == product_id) & (df['product_sold'] == product_sold)].copy()

        if len(product_subset) == 0:
            continue  # Skip if subset is empty

        # Get the value from the last row of 'customer_value' column
        value = product_subset['customer_value'].iloc[-1]

        # Features
        features = ['price', 'rating', 'total_rating', 'status']
        X = product_subset[features]

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
        product_subset['predicted_redistributed_value'] = weights_predicted * value

        # Calculate proportional redistributed value for each day
        product_subset['redistributed_value'] = weights * value
       
        # Save the predicted values back to the main DataFrame under 'customer_value' column
        df.loc[(df['product_id'] == product_id) & (df['product_sold'] == product_sold), 'customer_value'] = product_subset['predicted_redistributed_value']

    return df