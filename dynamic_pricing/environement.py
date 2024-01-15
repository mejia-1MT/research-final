import numpy as np 
import random 
import torch

# Define the PricingEnvironment
class PricingEnvironment:
    def __init__(self, data, num_products, state_size=7):
        self.data = data
        self.num_products = num_products
        self.reset()

    def reset(self):
        self.current_product = 0  # Start with the first product
        self.current_day = 0
        self.done = False

    def step(self, action):
        new_product_price = action  # Assuming 'action' is the price set by the agent for the new product
        state = self.get_state(new_product_price)  # Update state based on the new product and market conditions

        # Fetch relevant data for the current day
        product_data = self.data[self.data["product_id"] == self.current_product + 1]
        if self.current_day < len(product_data):
            current_row = product_data.iloc[self.current_day]

            # Get the price and other relevant information for the current day
            current_price = current_row["price"]
            customers_modifier = self.calculate_customers_modifier(current_price, new_product_price)

            # Update the number of customers based on the pricing decision and market conditions
            self.current_customers = int(current_row["product_sold"] * customers_modifier)
            self.data.at[self.current_day, "product_sold"] = self.current_customers

            reward = self.calculate_reward(new_product_price, self.current_customers)  # Update the calculate_reward method accordingly
        else:
            reward = 0

        self.current_day += 1
        if self.current_day == len(self.data):
            self.done = True

        return state, reward, self.done

    def get_state(self, new_product_price):
        # For example, assuming the state is a vector of relevant features for pricing:
        product_data = self.data[self.data["product_id"] == self.current_product + 1]  # Assuming product_id starts from 1

        # Initialize the state with zeros
        state = np.zeros(self.state_size)

        if self.current_day < len(product_data):
            current_row = product_data.iloc[self.current_day]

            state[:6] = np.array([
                current_row["day"],
                current_row["product_sold"],
                current_row["price"],
                current_row["rating"],
                current_row["total_rating"],
                current_row["status"]
            ])

        # Ensure that the state has the correct shape for the neural network
        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)

        return state_tensor

    def calculate_customers_modifier(self, current_price, new_price, base_customers=5):
        price_difference = new_price - current_price
        sensitivity = 10 # Higher sensitivity for more variability
        base_modifier = base_customers / self.num_products / 100
        sigmoid_value =  0.5 / (25 + np.exp(-sensitivity * price_difference / current_price))
        random_factor = random.uniform(0.8, 1.2)  # Adjust the range for variability
        customers_modifier = base_modifier + sigmoid_value * random_factor
        return customers_modifier



    def calculate_reward(self, action, customers):
        product_data = self.data[self.data["product_id"] == self.current_product + 1]

        if self.current_day < len(product_data):
            current_row = product_data.iloc[self.current_day]

            target_price = current_row["price"]
            deviation_penalty = -abs(action - target_price)
            reward = customers + deviation_penalty

            # Debugging print statements

        else:
            reward = 0

        return reward
