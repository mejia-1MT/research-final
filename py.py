import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import random
from demand import predict_customer
from compute_customer import redistribute_customer_value, calculate_customer_value
from simulate_episode import simulate
# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the agent
class DQNAgent:
    def __init__(self, input_size, output_size, initial_price, memory_capacity=10000, batch_size=32, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = [] 
        self.model = DQN(input_size, output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.min_price = initial_price * 0.8  # 20% below initial price
        self.max_price = initial_price * 1.2  # 20% above initial price
        self.num_actions = 50
        self.price_range = np.linspace(self.min_price, self.max_price, self.num_actions)

    def choose_action(self, state):  

        if random.uniform(-1, 3) < self.epsilon:
            #print('explor')
            action = random.randint(0, self.num_actions - 1)  # Choose a random action

            price = self.pricing(action)
            return price
        else:
            #print('exploittttttt')
            # print('state: ', state)
            q_values = self.model(torch.Tensor(state))
            # print('q_values: ', q_values)
            action = torch.argmax(q_values).item()
        
            # print("qvalues: ",q_values )
            #lipat to sa training later
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            #print('epsilon value: ', self.epsilon)
            price = self.pricing(action)
            return price
        
    def pricing (self, action):
        # print("action before indexing:", action)
        granular_price = self.price_range[action]
        return granular_price
    
    def pricing_to_index(self, price):
        # Map price to index within the price range (nodes)
        index = int(((price - self.min_price) / (self.max_price - self.min_price)) * self.num_actions)
        # print('index: ', index)
        return index -1

    def remember(self, state, action, reward, next_state, done):
        # Add a transition to the replay memory
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)  # Remove the oldest experience if memory capacity is exceeded

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        counter = 0
        # Sample random mini-batch from the replay memory
        mini_batch = random.sample(self.memory, self.batch_size)
        #print(f'mini_batch {counter} ', mini_batch)
        counter += 1
        
        for state, action, reward, next_state, done in mini_batch:
            if next_state is not None:
                state = torch.Tensor(state)
                next_state = torch.Tensor(next_state)
                q_values = self.model(state)
                next_q_values = self.model(next_state)

                # print("state: ",state)
                # print("next state: ",next_state)
                # #  # Convert next_q_values to a numpy array and then back to a PyTorch tensor
                # print("next q_vlues: ", next_q_values)
                # print(type(next_q_values))
            
                next_q_values = torch.Tensor(next_q_values.detach().clone())
                # print("next q_vlues detached: ", next_q_values)
                # print(type(next_q_values))
                max_next_q_value = torch.max(next_q_values).item()
                # print("max: ", max_next_q_value)
                # print(type(max_next_q_value))
                reward_tensor = torch.tensor(reward)
                target = q_values.clone().detach()
                # print("target: ", target)
                # print(type(target))
                # print("reward ", reward)
                # print(type(reward))
                # print("reward ", reward_tensor)
                # print(type(reward_tensor))
                # print("self.gamma ", self.gamma)
                # print(type(self.gamma))
                # print("max_next_q_value ", max_next_q_value)
                # print(type(max_next_q_value))
                # print("action ", action)
                # print(type(action))
                action = self.pricing_to_index(action)
                # print("action ", action)
                # print(type(action))
                target[0][action] = reward_tensor + self.gamma * max_next_q_value * (not done)
                # print("target[0] ", target[0][action])
                # print(type(target[0][action]))
                self.optimizer.zero_grad()
                loss = self.criterion(q_values, target)
                loss.backward()
                self.optimizer.step()

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
            else: 
                pass
    # Modify the train method to use experience replay
    def train(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.replay()

# Define the environment
class PricingEnvironment:
    def __init__(self, dataset, unprocessed, initial_price):
        self.products_data = dataset
        self.unprocessed_pd = unprocessed
        self.initial_price = initial_price


        self.total_days = 30
        self.current_day = 0
        self.current_product_index = 0
        self.total_products = 25
        # print(f"df: {self.products_data}")

    def reset(self, current_day):
        self.current_day = current_day + 1
        self.current_product_index = 1

        
        state = self.products_data.loc[(self.products_data['product_id'] == self.current_product_index) & (self.products_data['day'] == self.current_day)]
        state = state.iloc[:, 2:].values  # Convert DataFrame subset to a NumPy array, excluding the first two columns
        return state

    def step(self, action):

        #print(f"Day: {self.current_day} Product: {self.current_product_index} ")
       
        
        # print("Product: ", self.current_product_index)
        # prod num >= 30
        if self.current_product_index >= self.total_products:
            done = True
            next_state = None
            reward = 0
            simulated_sales = 0 
            baseline_sales = 0 
            d_demand_val = 0
            s_demand_val = 0
            return next_state, reward, done, simulated_sales, baseline_sales, d_demand_val, s_demand_val

        next_state = self.products_data[(self.products_data['day'] == self.current_day) & (self.products_data['product_id'] == self.current_product_index + 1)]
        next_state = next_state.iloc[:, 2:].values  # Convert DataFrame subset to a NumPy array, excluding the first two columns

        self.current_product_index += 1  # Move this line here

        reward, simulated_sales, baseline_sales, d_demand_val, s_demand_val = self.calculate_reward(action)
        done = False
        return next_state, reward, done, simulated_sales, baseline_sales,d_demand_val, s_demand_val

    def calculate_reward(self, action):
        # print('action', action)
        
        # Predict demand value for both Dynamic and Static Price
        simulated_sales, d_demand_val = predict_customer(action, self.unprocessed_pd)
        baseline_sales, s_demand_val = predict_customer(self.initial_price, self.unprocessed_pd)
        # print()
        # print(f'Simulated Sales: {simulated_sales}')
        # print(f'dyanmic salea values: {d_demand_val}')
        # print(f'Baseline Sales: {baseline_sales}')
        # print(f'static sales values: {s_demand_val}')
        # Calculate the change in sales from the baseline
        #print(f"type: {type(simulated_sales)}")
        # Calculate reward which is the difference in sales 
        change_in_sales = simulated_sales - baseline_sales
        
        return change_in_sales, simulated_sales, baseline_sales, d_demand_val, s_demand_val
    
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    # print(f"df not scaled: {df}")
    # Normalize 'product_sold', 'price', 'rating', 'total_rating', 'customer'
    scaler = MinMaxScaler()
    columns_to_normalize = ['product_sold', 'price', 'rating', 'total_rating', 'status', 'customer']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

  
    #print(f"df scaled: {df}")
    return df

def random_action(baseline_price):
    # Define the range of possible price adjustments or actions
    price_adjustment_range = 0.1 * baseline_price  # 10% of the baseline price
    random_adjustment = random.uniform(-price_adjustment_range, price_adjustment_range)
    
    # Apply the random adjustment to the baseline price
    new_price = baseline_price + random_adjustment
    return new_price
   
file_path = './Shopee-Product-20oz-September-2023-FINAL.xlsx'

data = pd.read_excel(file_path)

assert all(col in data.columns for col in ["product_id", "day", "product_sold", "price", "rating", "total_rating", "status"])

for product_id in data['product_id'].unique():
    product_df = data[data['product_id'] == product_id]
    
    # Calculate customer value based on the rule
    customer_value = calculate_customer_value(product_df)
    
    # Add the 'customer_value' column to the DataFrame
    data.loc[data['product_id'] == product_id, 'customer_value'] = customer_value


with_p = redistribute_customer_value(data)
# print('with_p', with_p)

processed_data = preprocess_data(with_p.copy())


# print('processed', processed_data)

# Example usage
num_products = 25
num_days = 30
num_features = 8
cliped_features = 6
baseline_price = 1900
epsilon_start = 1.0
min_epsilon = 0.01
decay_rate = 0.99
env = PricingEnvironment(processed_data, with_p, baseline_price)
agent = DQNAgent(cliped_features, 50, baseline_price)

num_episodes = 10
save_frequency = 10

save_path = 'model/saved/DP_model.pth'

if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
    agent.model.load_state_dict(torch.load(save_path))
    print("Model loaded successfully!")
else:
    print("No model found or the model file is empty.")

simulate(env, agent)

#print(processed_data[processed_data['day'] == 1].to_string(index=False))

# for episode in range(1):
#     for day in range(num_days):
#         state = env.reset(day)
#         done = False
#         total_reward = 0
#         actions = []
#         while not done:
            
#             action = agent.choose_action(state)
#             next_state, reward, done = env.step(action)
#             # print("next state: ", next_state)
#             agent.train(state, action, reward, next_state, done)
#             state = next_state
#             total_reward += reward
        
#         actions.append(action)


#     print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

#     if (episode + 1) % save_frequency == 0:
#         torch.save(agent.model.state_dict(), save_path)
#         print(f"Model saved at episode {episode + 1}")



