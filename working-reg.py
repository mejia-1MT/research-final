import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_dataset(file_path, num_days=30):
    try:
        data = pd.read_excel(file_path)

        assert all(col in data.columns for col in ["product_id", "day", "product_sold", "price", "rating", "total_rating", "status"])

        return data 
    except Exception as e:
        print(f"Error getting dataset: {e}")
        return None
    
def calculate_customer_value(product_df):
    customer_value = []
    product_sold_values = product_df['product_sold'].tolist()

    # Check if product_sold remained constant throughout the 30-day record
    if len(set(product_sold_values)) == 1:
        if product_sold_values[0] > 1000:
            # Set customer value to 50 for the changing value
            customer_value.extend([0] * 29)
            customer_value.append(50)
        else:
            # If product_sold is not higher than 1000, set customer value to 0
            customer_value.extend([0] * 30)
    else:
        # If product_sold changed, set customer value to 0 for all days
        customer_value.extend([0] * 30)
        
        # Rule 2: Calculate customer value for the first change if product_sold for that day is higher than 1000
        first_change_index = next((i for i, v in enumerate(product_sold_values) if v != product_sold_values[0] and v > 1000), None)
        if first_change_index is not None and first_change_index <= 30:
            customer_value[first_change_index - 1] = (first_change_index / 30) * 100
            
            # Find subsequent changes and calculate customer value accordingly until reaching day 30 or another change
            current_index = first_change_index
            while current_index < 30:
                next_change_index = next((i for i in range(current_index + 1, 30) if product_sold_values[i] != product_sold_values[current_index] and product_sold_values[i] > 1000), None)
                if next_change_index is not None:
                    customer_value[next_change_index - 1] = ((next_change_index - current_index) / 30) * 100
                    current_index = next_change_index
                else:
                    break
            
            # Check for changes between day 1 and day 30
            if product_sold_values[0] != product_sold_values[29] and product_sold_values[29] > 1000:
                days_taken = 30 - current_index
                customer_value[29] = (days_taken / 30) * 100

         # Rule 3: Calculate customer value if product_sold is less than 1000
        for i in range(1, 30):
            if product_sold_values[i] <= 1000:
                customer_value[i] = product_sold_values[i] - product_sold_values[i - 1]

    return customer_value



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def redistribute_customer_value(df):
    product_ids = df[df['product_sold'] > 1000]['product_id'].unique()

    # Create a new column to store predicted customer values
    df['predicted_customer_value'] = df['customer_value']

    for product_id in product_ids:
        product_subset = df[(df['product_id'] == product_id) & (df['product_sold'] > 1000)].copy()

        # Get unique ranges of product_sold for the current product_id
        product_sold_ranges = product_subset['product_sold'].unique()

        for sold_range in product_sold_ranges:
            subset_by_sold_range = product_subset[product_subset['product_sold'] == sold_range].copy()

            # Get the customer_value from the last day of the subset
            designated_customer_value = subset_by_sold_range['customer_value'].iloc[-1]

            X = subset_by_sold_range[['day', 'price', 'rating', 'total_rating', 'status']]
            y = subset_by_sold_range['customer_value']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1)
            model.fit(X_train, y_train)

            # Store predicted customer values in the new column
            subset_by_sold_range.loc[:, 'predicted_customer_value'] = model.predict(X)

            # Scale predicted values to match the designated customer_value
            predicted_sum = subset_by_sold_range['predicted_customer_value'].sum()
            if predicted_sum != 0 and np.isfinite(predicted_sum):
                subset_ratio = designated_customer_value / predicted_sum
                subset_by_sold_range['predicted_customer_value'] *= subset_ratio

                # Introduce randomness and disperse predicted values throughout the subset
                max_value = 10  # Define the maximum value for dispersion
                min_value = 1   # Define the minimum value for dispersion
                num_days = len(subset_by_sold_range)
                dispersed_values = np.linspace(min_value, max_value, num_days)

                # Shuffle the values to further disperse them
                np.random.shuffle(dispersed_values)
                subset_by_sold_range['predicted_customer_value'] = dispersed_values.round().astype(int)

                # Update the original DataFrame with the adjusted predicted values for the specific subset
                df.loc[subset_by_sold_range.index, 'predicted_customer_value'] = subset_by_sold_range['predicted_customer_value']
            else:
                print(f"Skipping subset due to zero division or non-finite predicted sum.")

            print(f"Subset for Product ID {product_id} - Product Sold: {sold_range} - Designated Customer Value: {designated_customer_value}:")
            print(subset_by_sold_range)
    return df


# # Rule 1: Initialize the customer_value column with specific conditions

file_path = './Shopee-Product-20oz-September-2023-FINAL.xlsx'
df = load_dataset(file_path)


# Loop through each unique product_id
for product_id in df['product_id'].unique():
    product_df = df[df['product_id'] == product_id]
    
    # Calculate customer value based on the rule
    customer_value = calculate_customer_value(product_df)
    
    # Add the 'customer_value' column to the DataFrame
    df.loc[df['product_id'] == product_id, 'customer_value'] = customer_value

#print(df.to_string(index=False))

with_p = redistribute_customer_value(df)

print(with_p.to_string(index=False))
