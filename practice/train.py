
import torch

def train_agent(agent, env, num_episodes = 50, save_frequency = 10, save_path = 'model/saved/DP_model.pth'):


    for episode in range(num_episodes):
        for day in range(env.total_days):
            state = env.reset(day)
            done = False
            total_reward = 0
            actions = []
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _, _, _, _ = env.step(action)
                agent.train(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                actions.append(action)

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

            if (episode + 1) % save_frequency == 0:
                torch.save(agent.model.state_dict(), save_path)
            print(f"Model saved at episode {episode + 1}")