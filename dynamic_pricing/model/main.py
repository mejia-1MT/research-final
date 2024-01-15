import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import MinMaxScaler
from data_prep import load_dataset
from compute_customer import redistribute_customer_value, calculate_customer_value
# from compute_combined_forecast import demand_forecast
from environment import ProductPricingEnvironment
from dqn_model import DQN
from modes.train import train
from modes.evaluate import evaluate
from modes.simulate_episode import simulate_episode



def main(mode):
    
    file_path = './Shopee-Product-20oz-September-2023-FINAL.xlsx'
    data = load_dataset(file_path)

    # Step 1: Data Preparation
    if data is not None:

        # Loop through each unique product_id
        #add this to data_prep
        for product_id in data['product_id'].unique():
            product_df = data[data['product_id'] == product_id]
            
            # Calculate customer value based on the rule
            customer_value = calculate_customer_value(product_df)
            
            # Add the 'customer_value' column to the DataFrame
            data.loc[data['product_id'] == product_id, 'customer_value'] = customer_value


        with_p = redistribute_customer_value(data)
       

        #print(len(with_p))

        # Hyperparameters
        initial_price = 900
        num_products = 25
        scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Initialize environment and DQN
        env = ProductPricingEnvironment(with_p, initial_price)  # Replace 'your_product_data' with your actual data
        input_dim = env.feature_dim
        output_dim = 1  # Outputting a single value (price)

        model = DQN(input_dim, output_dim)
        #model.load_state_dict(torch.load('model/modes/DP_model.pth')) 
        optimizer = optim.Adam(model.parameters(), lr=0.001 )
        criterion = nn.MSELoss()

        if mode == 'train':
            train(env, model, optimizer, criterion, scaler, num_products)
        elif mode == 'eval':
            evaluate(env, model, scaler)
        elif mode == 'simulate':
            simulate_episode(env, model, scaler)
        else:
            print("Invalid mode selected.")
    else:
        print("Failed to load the dataset.")
    
    
if __name__ == "__main__":
     # Choose your mode here: 'train', 'eval', or 'simulate'
    selected_mode = 'train'
    main(selected_mode)
