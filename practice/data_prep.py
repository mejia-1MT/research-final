import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from compute_customer import redistribute_customer_value, calculate_customer_value


def load_dataset(file_path):
    try:
        data = pd.read_excel(file_path)

        assert all(col in data.columns for col in ["product_id", "day", "product_sold", "price", "rating", "total_rating", "status"])

        # print("this is dataprep 1:", data)

        for product_id in data['product_id'].unique():
            product_df = data[data['product_id'] == product_id]
            
            # Calculate customer value based on the rule
            customer_value = calculate_customer_value(product_df)
            
            # Add the 'customer_value' column to the DataFrame
            data.loc[data['product_id'] == product_id, 'customer_value'] = customer_value

        with_p = redistribute_customer_value(data)

        print("this is dataprep 3:", data.to_string(index=False))
        return data
    
    except Exception as e:
        print(f"Error getting dataset: {e}")
        return None

def preprocess_data(df):
    # print(f"df not scaled: {df}")
    # Normalize 'product_sold', 'price', 'rating', 'total_rating', 'customer'
    scaler = MinMaxScaler()
    columns_to_normalize = ['product_sold', 'price', 'rating', 'total_rating', 'status', 'customer']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    # Get the scaling factors for each feature
    scaling_factors = scaler.scale_
    # print("Scaling factors for each feature:", scaling_factors)
    # print(f"df scaled: {df}")
    return df, scaling_factors

def scale_aggregated_state(aggregated_state):

    # print('aggregated_state in scaling ', aggregated_state)
    # Convert aggregated_state to numpy array for compatibility with MinMaxScaler
    aggregated_state_array = np.array(aggregated_state)
    
    # Create a MinMaxScaler instance
    scaler = MinMaxScaler()
    
    # Fit the scaler to the aggregated_state data and transform it
    scaled_state = scaler.fit_transform(aggregated_state_array)
    
    return scaled_state

def reverse_scaling(scaled_value, scaling_factors):
    # Reverse scaling using the scaling factor
    unscaled_value = scaled_value / scaling_factors
    return unscaled_value