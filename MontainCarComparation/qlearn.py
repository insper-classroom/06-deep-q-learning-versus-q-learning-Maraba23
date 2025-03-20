import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

# Configuração única de hiperparâmetros
learning_rate = 0.1
discount_factor = 0.99
epsilon_decay = 0.999
epsilon_min = 0.1

# Configurações do treinamento
num_executions = 10
episodes = 5000
num_bins = (20, 20)

env = gym.make('MountainCar-v0')

def discretize_state(state, bins):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / bins
    return tuple(((state - env_low) / env_dx).astype(int))

all_rewards = []

q_table = np.zeros(num_bins + (env.action_space.n,))

for execution in range(num_executions):
    rewards_per_episode = []
    epsilon = 1.0

    for episode in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state, num_bins)
        done = False
        total_reward = 0

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state, num_bins)

            q_table[state + (action,)] = (1 - learning_rate) * q_table[state + (action,)] + learning_rate * (
                reward + discount_factor * np.max(q_table[next_state])
            )

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        print(f"Execução: {execution + 1}/{num_executions} - Episódio: {episode + 1}/{episodes} - Recompensa: {total_reward}", end="\r")

    all_rewards.append(rewards_per_episode)

env.close()

# Salvar Q-table final
np.save("q_table_mountaincar.npy", q_table)
print("\nQ-Table salva como q_table_mountaincar.npy")

# Visualização
df_rewards = pd.DataFrame(all_rewards).T
df_rewards.columns = [f'Execução {i+1}' for i in range(num_executions)]
df_rewards["Média"] = df_rewards.mean(axis=1)

plt.figure(figsize=(10, 5))
plt.plot(df_rewards["Média"], label="Recompensa Média Móvel")
plt.xlabel("Episódio")
plt.ylabel("Recompensa Média")
plt.title("Q-Learning no MountainCar-v0")
plt.legend()
plt.grid()
plt.savefig("qlearning_mountaincar.png")
plt.show()

print("Gráfico salvo como qlearning_mountaincar.png")
