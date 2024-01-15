import random 
import numpy as np
from data_prep import scale_aggregated_state
# from plot import display_prices_graph

def simulate(env, agent):
    num_products = env.total_products
    num_days = env.total_days

    for day in range(1, num_days + 1)[:1]:
        aggregated_state = []

        for product_id in range(1, num_products + 1):
            state = env.get_product_state(product_id, day)  # Get state for product on this day
            print(f'state {state}')
            aggregated_state.extend(state)  # Aggregate state for all products

        #print(f'aggregated state {aggregated_state}')
        scaled_state = min_max_scaling(aggregated_state)
        #print('withoutminmax', scaled_state)
        scaled_states = scale_aggregated_state(aggregated_state)
        print('withminmax', scaled_states)
        done = False
        rewards = []
        actions = []
        d_demand = []
        s_demand = []
        d_demand_val = []
        s_demand_val = []
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, dynamic_demand, static_demand, d_demand_vals, s_demand_vals = env.step(action)
            state = next_state

            actions.append(action)
            d_demand.append(dynamic_demand)
            s_demand.append(static_demand)
            rewards.append(reward)
            s_demand_val.append(s_demand_vals)
            d_demand_val.append(d_demand_vals)
        # Get the index of the instance with highest demand
        highest_index = d_demand.index(max(d_demand))
      
    # num_products = env.total_products
    # num_days = env.total_days

    # for day in range(num_days):
    #     aggregated_state = []  # Aggregated state for all products on this day

    #     # Gather states for all products for this day
    #     for product_id in range(1, num_products + 1):
    #         state = env.get_product_state(product_id, day)  # Get state for product on this day
    #         aggregated_state.extend(state)  # Aggregate state for all products

    #     done = False
    #     rewards = []
    #     actions = []
    #     d_demand = []
    #     s_demand = []
    #     d_demand_val = []
    #     s_demand_val = []

    #     while not done:
    #         # Use the aggregated state for all products as input to the agent
    #         action = agent.choose_action(aggregated_state)

    #         # Apply the saaroducts for this day
    #         for product_id in range(1, num_products + 1):
    #             _, reward, _, dynamic_demand, static_demand, d_demand_vals, s_demand_vals = env.step(product_id, day, action)
    #             rewards.append(reward)
    #             d_demand.append(dynamic_demand)
    #             s_demand.append(static_demand)
    #             d_demand_val.append(d_demand_vals)
    #             s_demand_val.append(s_demand_vals)

    #         actions.append(action)
            
    #         # Update the aggregated state after taking the collective action for all products
    #         aggregated_state = []  # Reset the aggregated state
    #         for product_id in range(1, num_products + 1):
    #             state = env.get_product_state(product_id, day)  # Get updated state for each product
    #             aggregated_state.extend(state)  # Aggregate the updated state for all products

    #         # Check termination condition or end of the day
    #         done = env.check_termination_condition()

    #     # Perform any evaluation or updates at the end of the day based on rewards, demands, etc.
    #     # Update the agent based on the collected experiences from the day
    #     agent.update(rewards, actions, d_demand, s_demand, d_demand_val, s_demand_val)
















# def simulate(env, agent):
#     index_mapping = {}

#     for day in range(30):
#         state = env.reset(day)
#         done = False
#         rewards = []
#         actions = []
#         d_demand = []
#         s_demand = []
#         d_demand_val = []
#         s_demand_val = []
#         while not done:
#             action = agent.choose_action(state)
#             next_state, reward, done, dynamic_demand, static_demand, d_demand_vals, s_demand_vals = env.step(action)
#             state = next_state

#             actions.append(action)
#             d_demand.append(dynamic_demand)
#             s_demand.append(static_demand)
#             rewards.append(reward)
#             s_demand_val.append(s_demand_vals)
#             d_demand_val.append(d_demand_vals)
#         # Get the index of the instance with highest demand
#         highest_index = d_demand.index(max(d_demand))
      
#         index_mapping[day] = (
#             format(float(d_demand[highest_index]), '.2f'),
#             format(float(actions[highest_index]), '.2f'),
#             format(float(s_demand[highest_index]), '.2f'),
#             format(float(s_demand_val[highest_index]), '.2f'),
#             format(float(d_demand_val[highest_index]), '.2f'))

        
#         mapped_values = index_mapping[day]  # Retrieve newly mapped values

#         # print(f'Day {day+1} Price {mapped_values[1]} Demand Value {mapped_values[4]} Revenue {mapped_values[0]}')
#         # print(f'Day {day+1} Price {env.initial_price} Demand Value {mapped_values[3]} Revenue {mapped_values[2]}')
    
#     d_demand_list = []
#     actions_list = []
#     s_demand_list = []
#     s_demand_val_list = []
#     d_demand_val_list = []

#     for index, values in index_mapping.items():
#         d_demand, actions, s_demand, s_demand_val, d_demand_val = values
#         d_demand_list.append(d_demand)
#         actions_list.append(actions)
#         s_demand_list.append(s_demand)
#         s_demand_val_list.append(s_demand_val)
#         d_demand_val_list.append(d_demand_val)
    
#     # print(len(actions_list))

#     print("\n\nDynamic Pricing Breakdown\n")
#     min_revenue_dynamic = float('inf')
#     max_revenue_dynamic = float('-inf')
#     day_min_revenue_dynamic = -1
#     day_max_revenue_dynamic = -1
#     dynamic_daily_revenues = []

#     for i in range(len(actions_list)):
#         revenue_dynamic = d_demand_list[i]
#         print(f'Day {i+1} Price {actions_list[i]} Demand Value {d_demand_val_list[i]} Revenue {d_demand_list[i]}')
#         # Convert revenue_dynamic to float if it's a string
#         if isinstance(revenue_dynamic, str):
#             try:
#                 revenue_dynamic = float(revenue_dynamic)
#             except ValueError:
#                 print(f"Value '{revenue_dynamic}' cannot be converted to float.")
#                 continue  # Skip this iteration if conversion fails

#         dynamic_daily_revenues.append(revenue_dynamic)

#         # Track minimum and maximum revenue for dynamic pricing
#         if isinstance(revenue_dynamic, float):  # Ensure revenue_dynamic is a float for comparison
#             if revenue_dynamic < min_revenue_dynamic:
#                 min_revenue_dynamic = revenue_dynamic
#                 day_min_revenue_dynamic = i + 1
#             if revenue_dynamic > max_revenue_dynamic:
#                 max_revenue_dynamic = revenue_dynamic
#                 day_max_revenue_dynamic = i + 1
#     # and their corresponding days for both pricing methods
    
#     print("\n\nStatic Pricing Breakdown")
#     min_revenue_static = float('inf')
#     max_revenue_static = float('-inf')
#     day_min_revenue_static = -1
#     day_max_revenue_static = -1
#     static_daily_revenues = []

#     for i in range(len(actions_list)):
#         demand_val = float(d_demand_val_list[i])
#         revenue_static = env.initial_price * demand_val
#         print(f'Day {i+1} Price {env.initial_price} Demand Value {d_demand_val_list[i]} Revenue {revenue_static}')

#         # Convert revenue_static to float if it's a string
#         if isinstance(revenue_static, str):
#             try:
#                 revenue_static = float(revenue_static)
#             except ValueError:
#                 print(f"Value '{revenue_static}' cannot be converted to float.")
#                 continue  # Skip this iteration if conversion fails

#         static_daily_revenues.append(revenue_static)

#         # Track minimum and maximum revenue for static pricing
#         if isinstance(revenue_static, float):  # Ensure revenue_static is a float for comparison
#             if revenue_static < min_revenue_static:
#                 min_revenue_static = revenue_static
#                 day_min_revenue_static = i + 1
#             if revenue_static > max_revenue_static:
#                 max_revenue_static = revenue_static
#                 day_max_revenue_static = i + 1


#     total_revenue_dynamic = sum(dynamic_daily_revenues)
#     total_revenue_static = sum(static_daily_revenues)
   

#     prices_as_floats = [float(price) for price in actions_list]

    
#     print("\n\n********************************************\n\n")
#     print("Analysis:\n")
#     print("\n\nHighest and Lowest Revenue Days for both strategy\n")
#     print("Dynamic Pricing - Minimum Revenue:", min_revenue_dynamic, "on Day:", day_min_revenue_dynamic)
#     print("Dynamic Pricing - Maximum Revenue:", max_revenue_dynamic, "on Day:", day_max_revenue_dynamic)
    
#     print("\nStatic Pricing - Minimum Revenue:", min_revenue_static, "on Day:", day_min_revenue_static)
#     print("Static Pricing - Maximum Revenue:", max_revenue_static, "on Day:", day_max_revenue_static)
    

#     print("\n\nAssessment on revenues\n")
#     print(f"Total Revenue for Dynamic Pricing: {total_revenue_dynamic:.2f}")  # Print total revenue for dynamic pricing
#     print(f"Total Revenue for Static Pricing:: {total_revenue_static:.2f}")  # Print total revenue for dynamic pricing


#     dynamic_std_dev = np.std(dynamic_daily_revenues)
#     static_std_dev = np.std(static_daily_revenues)

#     print(f"Standard deviation of Dynamic Pricing: {dynamic_std_dev:.2f}")
#     print(f"Standard deviation of Static Pricing: {static_std_dev:.2f}")
    
#     static_daily_revenues = np.array(static_daily_revenues)
#     dynamic_daily_revenues = np.array(dynamic_daily_revenues)
#     # Calculate Mean Absolute Error (MAE)
#     mae = np.mean(np.abs(static_daily_revenues - dynamic_daily_revenues))

#     # Calculate Root Mean Squared Error (RMSE)
#     rmse = np.sqrt(np.mean((static_daily_revenues - dynamic_daily_revenues) ** 2))

#     # Calculate Margin of Error (MOE) as percentage difference
#     moe = np.mean((dynamic_daily_revenues - static_daily_revenues) / static_daily_revenues) * 100
#     print("\n\nAccuracy of Dynamic compared to Static\n")
#     print(f"Mean Absolute Error (MAE): {mae:.2f}")
#     print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
#     print(f"Margin of Error (MOE): {moe:.2f}%")

#     # print("Dynamic Price and Demand")
#     # for i in range (len(actions_list)):
#     #     print(f"{actions_list[i]} {d_demand_val_list[i]}")
        

    
#     from sklearn.linear_model import LinearRegression
#     # Assuming 'prices' and 'demands' contain your price and demand data
#     price_str = actions_list
#     demand_str = d_demand_val_list

#     price = [float(val) for val in price_str if val.replace('.', '', 1).isdigit()]
#     demand = [float(val) for val in demand_str if val.replace('.', '', 1).isdigit()]

#     for val in price:
#         if not isinstance(val, (int, float)):
#             print(f"Non-numeric value found in price: {val}")
#             print(f'val dataype: {type(val)}')

#     # Check for non-numeric values in demand
#     for val in demand:
#         if not isinstance(val, (int, float)):
#             print(f"Non-numeric value found in demand: {val}")
#             print(f'val dataype: {type(val)}')

#     # Convert lists to NumPy arrays
#     demand = np.array(demand)
#     price = np.array(price)

#     demand = demand.reshape(-1, 1)  # Reshape to a column vector
#     price = price.reshape(-1, 1)  # Reshape to a column vector

#     # Step 1: Predict Demand for the 31st day
#     model_demand = LinearRegression()
#     model_demand.fit(np.arange(1, len(demand) + 1).reshape(-1, 1), demand)  # Fit the model

#     print("\n\n____________________________________________\n\n")
#     predicted_demand_31 = model_demand.predict(np.array([[31]]))  # Predict the demand for day 31
#     predicted_demand_31 = predicted_demand_31[0][0]  # Extracting the predicted demand value
#     print(f"Predicted demand for the 31st day: {predicted_demand_31:.2f}")

#     # Step 2: Predict Price for the Predicted Demand
#     model_price = LinearRegression()
#     model_price.fit(demand, price)  # Fit the model

#     predicted_price_31 = model_price.predict(np.array([[predicted_demand_31]]))  # Predict price for the predicted demand
#     predicted_price_31 = predicted_price_31[0][0]  # Extracting the predicted price value
#     print(f"Recommended price for the 31st day based on demand: {predicted_price_31:.2f}\n\n")

#     # display_prices_graph(prices_as_floats, env.initial_price, dynamic_daily_revenues, static_daily_revenues,
#     #     d_demand_val_list, predicted_price_31, predicted_demand_31, total_revenue_dynamic, total_revenue_static)