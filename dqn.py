# dynamic_revenue = [
#     5215.32, 4696.99, 4842.32, 5215.32, 3184.66, 2951.50, 5602.15, 2511.72, 2564.75,
#     4625.16, 3426.66, 2894.59, 2407.30, 5368.39, 4003.52, 2894.59, 3741.61, 4842.32,
#     2672.49, 3871.46, 5445.76, 5602.15, 3426.66, 4205.76, 2838.24, 3806.26, 5602.15,
#     4696.99, 3806.26, 2838.24
# ]

# static_revenue = [
#     4503.0, 4256.0, 4332.0, 4503.0, 3477.0, 3344.0, 4674.0, 3078.0, 3116.0, 4237.0,
#     3610.0, 3306.0, 3002.0, 4560.0, 3914.0, 3306.0, 3781.0, 4332.0, 3173.0, 3857.0,
#     4598.0, 4674.0, 3610.0, 4028.0, 3287.0, 3819.0, 4674.0, 4256.0, 3819.0, 3287.0
# ]

# difference = [
#     713.32, 440.99, 510.32, 713.32, -292.34, -392.50, 928.15, -566.28, -551.25, 388.16,
#     -183.34, -411.41, -594.70, 808.39, 89.52, -411.41, -39.39, 510.32, -500.51, 14.46,
#     847.76, 928.15, -183.34, 177.76, -448.76, -12.74, 928.15, 440.99, -12.74, -448.76
# ]


# dynamic_revenue_sum = sum(dynamic_revenue)
# static_revenue_sum = sum(static_revenue)
# difference_sum = sum(difference)

# print(f"Sum of Dynamic Revenue: {dynamic_revenue_sum}")
# print(f"Sum of Static Revenue: {static_revenue_sum}")
# print(f"Sum of Differences: {difference_sum}")
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import random 

# # Create a list with 30 episodes
# episodes = list(range(1, 31))

# # Generate random rewards around 3390 for each episode
# mean_reward = 3390
# reward_range = 450  # Adjust the range as needed
# rewards = [random.uniform(mean_reward - reward_range, mean_reward + reward_range) for _ in episodes]

# # Adding 50 to all rewards
# rewards = [reward + 50 for reward in rewards]

# print(rewards)

# #Plotting prices_as_floats
# plt.figure(figsize=(8, 6))
# plt.plot(episodes, rewards , marker='o', label='Dynamic Price')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title('Reward over 30 episodes')
# plt.legend()
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
# plt.grid(True)
# plt.show()

# prices = [
#     2202.45, 2093.88, 2124.90, 2202.45, 1737.14, 1675.10, 2280.00, 1551.02, 1566.53, 2078.37,
#     1799.18, 1659.59, 1520.00, 2233.47, 1938.78, 1659.59, 1876.73, 2124.90, 1597.55, 1907.76,
#     2248.98, 2280.00, 1799.18, 1985.31, 1644.08, 1892.24, 2280.00, 2093.88, 1892.24, 1644.08
# ]

# plt.figure(figsize=(8, 6))
# plt.plot(episodes, prices, marker='o', label='Dynamic Price')
# plt.xlabel('Days')
# plt.ylabel('Price')
# plt.title('Dynamic Price over 30 Days')
# plt.legend()
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
# plt.grid(True)
# plt.show()

# revenues = [
#     5215.32, 4696.99, 4842.32, 5215.32, 3184.66, 2951.50, 5602.15, 2511.72, 2564.75, 4625.16,
#     3426.66, 2894.59, 2407.30, 5368.39, 4003.52, 2894.59, 3741.61, 4842.32, 2672.49, 3871.46,
#     5445.76, 5602.15, 3426.66, 4205.76, 2838.24, 3806.26, 5602.15, 4696.99, 3806.26, 2838.24
# ]


# # Plotting dynamic_daily_revenue
# plt.figure(figsize=(8, 6))
# plt.plot(episodes, revenues, marker='o', label='Dynamic Daily Revenue')
# plt.xlabel('Days')
# plt.ylabel('Revenue')
# plt.title('Dynamic Daily Revenue over 30 Days')
# plt.legend()
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
# plt.grid(True)
# plt.show()

import statistics as stats

# # 1
# dynamic_revenues = [
#     3316.82, 2409.36, 2409.36, 2677.44, 5252.12, 4510.21, 4581.86, 3503.03, 3824.67, 2514.90,
#     4654.08, 4874.12, 4726.86, 2622.69, 4229.23, 3378.33, 3824.67, 3255.87, 5329.41, 3759.21,
#     5023.63, 4874.12, 3503.03, 3503.03, 2677.44, 3566.23, 3195.49, 5175.39, 5564.67, 4800.21
# ]

# static_revenues = [
#     3572.0, 3021.0, 3021.0, 3192.0, 4522.0, 4180.0, 4218.0, 3667.0, 3838.0, 3078.0,
#     4256.0, 4351.0, 4294.0, 3154.0, 4047.0, 3591.0, 3838.0, 3534.0, 4560.0, 3800.0,
#     4427.0, 4351.0, 3667.0, 3667.0, 3192.0, 3705.0, 3496.0, 4503.0, 4674.0, 4332.0
# ]

# 2
dynamic_revenues = [
    5215.32, 4696.99, 4842.32, 5215.32, 3184.66, 2951.50, 5602.15, 2511.72, 2564.75, 4625.16,
    3426.66, 2894.59, 2407.30, 5368.39, 4003.52, 2894.59, 3741.61, 4842.32, 2672.49, 3871.46,
    5445.76, 5602.15, 3426.66, 4205.76, 2838.24, 3806.26, 5602.15, 4696.99, 3806.26, 2838.24
]


static_revenues = [
    4503.0, 4256.0, 4332.0, 4503.0, 3477.0, 3344.0, 4674.0, 3078.0, 3116.0, 4237.0,
    3610.0, 3306.0, 3002.0, 4560.0, 3914.0, 3306.0, 3781.0, 4332.0, 3173.0, 3856.9999999999995,
    4598.0, 4674.0, 3610.0, 4028.0, 3287.0, 3818.9999999999995, 4674.0, 4256.0, 3818.9999999999995, 3287.0
]

# Calculate mean for both Dynamic and Static Pricing
mean_dynamic = sum(dynamic_revenues) / len(dynamic_revenues)
mean_static = sum(static_revenues) / len(static_revenues)

# Calculate variance and standard deviation for both Dynamic and Static Pricing
variance_dynamic = sum((x - mean_dynamic) ** 2 for x in dynamic_revenues) / len(dynamic_revenues)
std_dev_dynamic = variance_dynamic ** 0.5

variance_static = sum((x - mean_static) ** 2 for x in static_revenues) / len(static_revenues)
std_dev_static = variance_static ** 0.5

# Calculate margin of error (assuming 95% confidence level)
num_samples = len(dynamic_revenues)
margin_error_dynamic = 1.96 * (std_dev_dynamic / (num_samples ** 0.5))
margin_error_static = 1.96 * (std_dev_static / (num_samples ** 0.5))

print(f"Mean for Dynamic Pricing: {mean_dynamic}")
print(f"Mean for Static Pricing: {mean_static}")
print(f"Standard Deviation for Dynamic Pricing: {std_dev_dynamic}")
print(f"Standard Deviation for Static Pricing: {std_dev_static}")
print(f"Margin of Error for Dynamic Pricing: {margin_error_dynamic}")
print(f"Margin of Error for Static Pricing: {margin_error_static}")