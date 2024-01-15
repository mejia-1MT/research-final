from demand import predict_customer

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

        print(f"Day: {self.current_day} Product: {self.current_product_index} ")

        self.current_day = current_day + 1
        self.current_product_index = 1

        
        state = self.products_data.loc[(self.products_data['product_id'] == self.current_product_index) & (self.products_data['day'] == self.current_day)]
        state = state.iloc[:, 2:].values  # Convert DataFrame subset to a NumPy array, excluding the first two columns
        return state

    def step(self, action):

        #print(f"Day: {self.current_day} Product: {self.current_product_index} ")

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

        #print(f'reward: {reward}')
        return next_state, reward, done, simulated_sales, baseline_sales, d_demand_val, s_demand_val

    def calculate_reward(self, action):
        # print('action', action)
        
        # Predict demand value for both Dynamic and Static Price
        simulated_sales, d_demand_val = predict_customer(action, self.unprocessed_pd)
        baseline_sales, s_demand_val = predict_customer(self.initial_price, self.unprocessed_pd)
        # print()
        # print(f'Simulated Sales: {simulated_sales}')
        # print(f'dyanmic salea values: {d_demand_val}')
        # print(f'Baseline Sales: {baseline_sales}')
        # print(f'static sales values: {s_demand_val}')+
        # Calculate the change in sales from the baseline
        #print(f"type: {type(simulated_sales)}")
        # Calculate reward which is the difference in sales 
        change_in_sales = simulated_sales - baseline_sales

        
        return change_in_sales, simulated_sales, baseline_sales, d_demand_val, s_demand_val
    
