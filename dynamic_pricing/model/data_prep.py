import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from compute_customer import redistribute_customer_value, calculate_customer_value


def load_dataset(file_path):
    try:
        data = pd.read_excel(file_path)
        #print("data ", data)

        assert all(col in data.columns for col in ["product_id", "day", "product_sold", "price", "rating", "total_rating", "status"])


        for product_id in data['product_id'].unique():
            product_df = data[data['product_id'] == product_id]
            
            # Calculate customer value based on the rule
            customer_value = calculate_customer_value(product_df)
            
            # Add the 'customer_value' column to the DataFrame
            data.loc[data['product_id'] == product_id, 'customer_value'] = customer_value

        print("data ", data)
        with_p = redistribute_customer_value(data)


        return with_p
    
    except Exception as e:
        print(f"Error getting dataset: {e}")
        return None

def preprocess_data(df):
    # print(f"df not scaled: {df}")
    # Normalize 'product_sold', 'price', 'rating', 'total_rating', 'customer'
    scaler = MinMaxScaler()
    columns_to_normalize = ['product_sold', 'price', 'rating', 'total_rating', 'status', 'customer']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df
