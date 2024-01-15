import numpy as np

# Dynamic Pricing Breakdown data (prices)
dynamic_revenues = [
    3316.82, 2409.36, 2409.36, 2677.44, 5252.12, 4510.21, 4581.86, 3503.03, 3824.67, 2514.90,
    4654.08, 4874.12, 4726.86, 2622.69, 4229.23, 3378.33, 3824.67, 3255.87, 5329.41, 3759.21,
    5023.63, 4874.12, 3503.03, 3503.03, 2677.44, 3566.23, 3195.49, 5175.39, 5564.67, 4800.21
]

# Calculate the mean (average) revenue for the dynamic breakdown
mean_dynamic_revenue = sum(dynamic_revenues) / len(dynamic_revenues)
print("Mean Revenue for Dynamic Pricing Breakdown:", mean_dynamic_revenue)

population_std_dynamic_revenue = np.std(dynamic_revenues)
print("Population Standard Deviation for Dynamic Revenue:", population_std_dynamic_revenue)




# Static Pricing Breakdown data (prices)
static_revenues = [
    3572.0, 3021.0, 3021.0, 3192.0, 4522.0, 4180.0, 4218.0, 3667.0, 3838.0, 3078.0,
    4256.0, 4351.0, 4294.0, 3154.0, 4047.0, 3591.0, 3838.0, 3534.0, 4560.0, 3800.0,
    4427.0, 4351.0, 3667.0, 3667.0, 3192.0, 3705.0, 3496.0, 4503.0, 4674.0, 4332.0
]

# Calculate the mean (average) revenue for the static breakdown
mean_static_revenue = sum(static_revenues) / len(static_revenues)
print("Mean Revenue for Static Pricing Breakdown:", mean_static_revenue)

population_std_static_revenue = np.std(static_revenues)
print("Population Standard Deviation for Static Revenue:", population_std_static_revenue)



sample_size = 30  # Sample size for both dynamic and static breakdown

# Calculate MOE for dynamic breakdown
z_score = 1.96  # For a 95% confidence level
moe_dynamic = (population_std_dynamic_revenue / np.sqrt(sample_size)) * z_score
print("Margin of Error for Dynamic Pricing Breakdown:", moe_dynamic)

# Calculate MOE for static breakdown
moe_static = (population_std_static_revenue / np.sqrt(sample_size)) * z_score
print("Margin of Error for Static Pricing Breakdown:", moe_static)
