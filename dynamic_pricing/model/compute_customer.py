import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim



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


def redistribute_customer_value(df):

    unique_combinations = df[df['product_sold'] > 1000].groupby(['product_id', 'product_sold']).size().reset_index().drop(0, axis=1)
    
    df['customer'] = df['customer_value']

    for index, row in unique_combinations.iterrows():
        product_id = row['product_id']
        product_sold = row['product_sold']

        product_subset = df[(df['product_id'] == product_id) & (df['product_sold'] == product_sold)].copy()
        
        
        if len(product_subset) == 0:
            continue  # Skip if subset is empty

        # Extract the designated customer value before feature extraction
        designated_customer_value = product_subset['customer_value'].iloc[-1]

        # Set zeros where the designated customer value is found
        df.loc[(df['product_id'] == product_id) & (df['product_sold'] == product_sold), 'customer_value'] = 0

        # Initialize predicted_customer_value as 0 within the subset
        df.loc[(df['product_id'] == product_id) & (df['product_sold'] == product_sold), 'customer'] = 0

        # Extract features and target
        features = product_subset[['product_id', 'day', 'product_sold', 'price', 'rating', 'total_rating', 'status']]
        target = product_subset['customer_value']

        # Normalize features if needed (e.g., using MinMaxScaler from sklearn)

        # Split the data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
        # Ensure the target tensor has the right shape for MSE loss
        y_train_tensor = torch.unsqueeze(y_train_tensor, 1)
        y_val_tensor = torch.unsqueeze(y_val_tensor, 1)

        # Define the neural network model
        class CustomModel(nn.Module):
            def __init__(self):
                super(CustomModel, self).__init__()
                self.fc1 = nn.Linear(7, 64)  # Input size 7 (number of features), output size 64
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(64, 1)  # Output size 1 (predicted customer value)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # Initialize model and define loss function and optimizer
        model = CustomModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 1000
        batch_size = 32
        # Inside your training loop
        # Training loop
        for epoch in range(num_epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Validation
            model.eval()
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)  # Calculate validation loss
            #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        # Obtain predictions AFTER the training loop
        model.eval()
        predicted_customer_values = model(X_val_tensor).detach().numpy()
        predicted_customer_values = np.clip(predicted_customer_values, a_min=0, a_max=None)

        # Find the number of days with zero predicted customer value
        num_days_zero_value = (predicted_customer_values == 0).sum()

        if num_days_zero_value > 0:
            # Calculate the average value to redistribute among days with zero predicted customer value
            non_zero_values = predicted_customer_values[predicted_customer_values > 0]

            if len(non_zero_values) > 0:
                average_value = non_zero_values.sum() / len(non_zero_values)
            else:
                # Handle the case where there are no non-zero values
                average_value = 0  # Or any other appropriate value
            # Redistribute the average value among days with zero predicted customer value
            for i in range(len(predicted_customer_values)):
                if predicted_customer_values[i] == 0:
                    predicted_customer_values[i] = average_value

        # Redistribute predicted values to meet the designated customer value
        predicted_customer_values /= predicted_customer_values.sum()  # Normalize to sum to 1

        if np.isnan(predicted_customer_values).any() or predicted_customer_values.sum() == 0:
            # Handling NaN or zero sum by assigning equal values
            predicted_customer_values[:] = designated_customer_value / len(predicted_customer_values)

        predicted_customer_values *= designated_customer_value  # Scale to the designated customer value

        # Assign redistributed customer values to the original dataframe
        indices = X_val.index.tolist()
        for i, idx in enumerate(indices):
            df.at[idx, 'customer'] = predicted_customer_values[i][0]

        # Print subset details AFTER assigning predicted_customer_values
        designated_customer_value = target.iloc[-1]  # Designated customer value for the subset
        # print(f"Subset for Product ID {product_id} - Product Sold: {product_sold} - Designated Customer Value: {designated_customer_value}:")
        # print(df[(df['product_id'] == product_id) & (df['product_sold'] == product_sold)])  # Print the subset from the DataFrame

    return df.drop('customer_value', axis=1).round({'customer': 2})