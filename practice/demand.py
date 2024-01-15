from sklearn.linear_model import LinearRegression
import numpy as np
from data_prep import load_dataset
from joblib import dump, load
from compute_customer import redistribute_customer_value, calculate_customer_value

def train_model(data):
    # Extract price and customer from the dataset
    
    # print(f"price: {data['price']}")
    price = data['price'].tolist()
    customer = data['customer'].tolist()
    prices = np.array(price)
    customers = np.array(customer)

    # Reshape the price and customer data for sklearn's input
    prices = prices.reshape(-1, 1)
    customers = customers.reshape(-1, 1)

    # Train a simple linear regression model using both price and customer data
    model = LinearRegression()
    model.fit(prices, customers)
    return model

def save_model(model, filename='model/saved/sales_model.joblib'):
    dump(model, filename)

def load_model(filename='model/saved/sales_model.joblib'):
    return load(filename)

def predict_sales(model, baseline_price, average_customer_value):
    #print(data.to_string(index=False))
    new_data = np.array([[baseline_price]])
    predicted_customer = model.predict(new_data)

    # If the customer value for the new product is not available, use the average customer value
    new_customer_value = average_customer_value if np.isnan(predicted_customer) else predicted_customer

    # Sales prediction logic 
    predicted_sales = baseline_price * new_customer_value

    return predicted_sales, new_customer_value

def predict_customer(baseline_price, data, train=False):
    if train:
        model = train_model(data)
        save_model(model)  # Save the trained model
    else:
        model = load_model()  # Load the pre-trained model

    # Calculate the average customer value and take the average
    customer = data['customer'].tolist()
    # print(f"customer: {customer}")

    customers = np.array(customer)
    # print(f"customersum: {np.sum(customers)}")
    average_customer_value = np.mean(customers)
    # print("average customer: ", average_customer_value)

    # Perform sales prediction for the new product at the given baseline price
    predicted_sales = predict_sales(model, baseline_price, average_customer_value)

    # print("predicted_sale in demand: ",predicted_sales)
    
    return predicted_sales
# Example usage if you run this script di rectly

if __name__ == "__main__":

    file_path = './Shopee-Product-20oz-September-2023-FINAL.xlsx'
    data = load_dataset(file_path)
    # Loop through each unique product_id
    # add this to data_prep
    for product_id in data['product_id'].unique():
        product_df = data[data['product_id'] == product_id]
        
        # Calculate customer value based on the rule
        customer_value = calculate_customer_value(product_df)
        
        # Add the 'customer_value' column to the DataFrame
        data.loc[data['product_id'] == product_id, 'customer_value'] = customer_value


    with_p = redistribute_customer_value(data)
    #print(with_p.to_string(index=False))
    baseline_price = 1900
  
    result = predict_customer(baseline_price, with_p, train=False)
    print("Predicted sales for the new product at baseline price:", result)