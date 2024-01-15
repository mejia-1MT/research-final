import torch
import os
import pandas as pd
from environment import PricingEnvironment
from dqn_agent import DQNAgent
from simulate_episode import simulate
from train import train_agent
from data_prep import load_dataset, preprocess_data
import tkinter as tk
from tkinter import simpledialog
from tkinter import ttk

# def get_baseline_price():
#     root = tk.Tk()
#     root.lift()
#     root.title("Initial input for Dynamic Pricing")
#     # Calculate the screen width and height
#     screen_width = root.winfo_screenwidth()
#     screen_height = root.winfo_screenheight()

#     # Set the window dimensions
#     window_width = 300
#     window_height = 120

#     # Calculate the x and y positions for the window to be centered
#     x_position = (screen_width // 2) - (window_width // 2)
#     y_position = (screen_height // 2) - (window_height // 2)

#     root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

#     root.geometry("300x150")
    
    
#     frame = ttk.Frame(root)
#     frame.pack(pady=20)

#     label = ttk.Label(frame, text="Enter Baseline Price:")
#     label.grid(row=0, column=0, padx=10, pady=5)

#     baseline_price_entry = ttk.Entry(frame)
#     baseline_price_entry.grid(row=0, column=1, padx=10, pady=5)

#     def submit_price():
#         baseline_price = float(baseline_price_entry.get())
#         root.quit()
#         root.destroy()
#         if baseline_price is not None:
#             print('\nBaseline Price:', baseline_price)

#             # Example usage
#             num_features = 8
#             clipped_features = 6

#             env = PricingEnvironment(processed_data, data, baseline_price)
#             agent = DQNAgent(clipped_features, 50, baseline_price)
#             save_path = 'model/saved/DP_model.pth'

#             if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
#                 agent.model.load_state_dict(torch.load(save_path))
#                 print("\nModel loaded successfully!\n")
#             else:
#                 print("No model found or the model file is empty.")

#             #train(env, agent)
#             simulate(env, agent)
#         else:
#             print("Baseline Price input canceled or not provided.")

#     submit_button = ttk.Button(frame, text="Submit", command=submit_price)
#     submit_button.grid(row=1, columnspan=2, padx=10, pady=15)

#     root.mainloop()

# file_path = './Shopee-Product-20oz-September-2023-FINAL.xlsx'

# data = load_dataset(file_path)
# processed_data = preprocess_data(data.copy())

# get_baseline_price()


file_path = './Shopee-Product-20oz-September-2023-FINAL.xlsx'

data = load_dataset(file_path)
processed_data = preprocess_data(data.copy())


baseline_price = 1900  # Set the baseline price directly

# Remaining code stays the same
print('\nBaseline Price:', baseline_price)

# Example usage
num_features = 8
clipped_features = 6



env = PricingEnvironment(processed_data, data, baseline_price)
agent = DQNAgent(clipped_features, 50, baseline_price)
save_path = 'model/saved/DP_model.pth'

if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
    agent.model.load_state_dict(torch.load(save_path))
    print("\nModel loaded successfully!\n")
else:
    print("No model found or the model file is empty.")

# train(env, agent)
simulate(env, agent)
# Rest of the code remains unchanged
