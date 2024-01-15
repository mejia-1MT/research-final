import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

central_number = 117537
num_values = 30

# Generate 30 random values around the central number
reward = [random.gauss(central_number, 1000) for _ in range(num_values)]

print(reward)  # Print the generated random values
days = list(range(1, 31))  # Assuming 30 days

# Plotting env.initial_price
plt.figure(figsize=(8, 6))
plt.plot(days, reward, marker='o', label='Static Price')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over 30 episode')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
plt.grid(True)
plt.show()