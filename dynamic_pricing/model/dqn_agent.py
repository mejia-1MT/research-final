import torch
import torch.nn as nn
import numpy as np
import random
from dqn_model import DQN

# Define the DQN agent class
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

        if random.uniform(-1, 1) < self.epsilon:
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

