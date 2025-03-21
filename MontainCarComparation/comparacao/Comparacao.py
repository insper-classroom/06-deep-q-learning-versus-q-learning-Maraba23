import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import time
import os
from datetime import datetime

# Configurar estilo visual para os gráficos
plt.style.use('ggplot')
sns.set(style="whitegrid", font_scale=1.2)
colors = sns.color_palette("viridis", 10)

# Configurações globais
SEED = 42
EXECUTIONS = 5
EPISODES = 1500
SMOOTHING_WINDOW = 100  # Para suavizar as curvas de recompensa

# Configurar seeds para reprodutibilidade
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Criar diretório para salvar resultados
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

#========== IMPLEMENTAÇÃO DQN ==========
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
        # Melhor inicialização dos pesos
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Função para normalizar estados
def normalize_state(state):
    # Para MountainCar, position range é [-1.2, 0.6], velocity range é [-0.07, 0.07]
    position, velocity = state
    normalized_position = (position + 1.2) / 1.8  # Mapeia para [0, 1]
    normalized_velocity = (velocity + 0.07) / 0.14  # Mapeia para [0, 1]
    return np.array([normalized_position, normalized_velocity], dtype=np.float32)

# Função para modificar a recompensa (reward shaping)
def shape_reward(state, reward, done):
    position, velocity = state
    # Recompensa por atingir posições mais altas
    position_reward = (position + 1.2) / 1.8  # Normalizado para [0, 1]
    
    # Recompensa por ter velocidade (em qualquer direção)
    velocity_reward = abs(velocity) / 0.07  # Normalizado para [0, 1]
    
    # Dar maior peso para a combinação de posição alta e velocidade
    shaped_reward = position_reward + velocity_reward
    
    # Bônus extra para encorajar o movimento correto nas encostas
    if (position < -0.5 and velocity > 0) or (position > -0.1 and velocity < 0):
        shaped_reward += 0.5
    
    # Se terminou (alcançou o objetivo), adicione um grande bônus
    if done and position >= 0.5:
        shaped_reward += 10
        
    return shaped_reward

class DeepQLearning:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_dec=0.9998, 
                 batch_size=64, memory_size=100000, max_steps=200, 
                 lr=0.001, target_update_freq=100, device="cpu"):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.max_steps = max_steps
        self.device = device
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        self.model = DQN(input_dim, output_dim).to(self.device)
        self.target_model = DQN(input_dim, output_dim).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        state = normalize_state(state)
        
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def experience(self, state, action, reward, next_state, done):
        state = normalize_state(state)
        next_state = normalize_state(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions).squeeze()
        
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze()
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec
            
        return loss.item()

#========== IMPLEMENTAÇÃO Q-LEARNING ==========
def discretize_state(state, env, bins):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / bins
    return tuple(((state - env_low) / env_dx).astype(int))

#========== FUNÇÕES DE EXECUÇÃO ==========
def run_dqn(episodes=EPISODES, seed=None):
    """Executa o algoritmo DQN por um número específico de episódios"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    env = gym.make('MountainCar-v0')
    if seed is not None:
        env.reset(seed=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = DeepQLearning(
        env=env, 
        device=device,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_dec=0.999,  # Ajustado para 5000 episódios
        batch_size=64,
        memory_size=100000,
        max_steps=2000,
        lr=0.001,
        target_update_freq=100
    )
    
    rewards_history = []
    steps_since_update = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(agent.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Reward shaping
            shaped_reward = shape_reward(state, reward, done)
            
            agent.experience(state, action, shaped_reward, next_state, done)
            total_reward += reward  # Guardamos a recompensa original para comparação
            
            # Múltiplas atualizações por passo
            for _ in range(4):
                loss = agent.experience_replay()
            
            steps_since_update += 1
            agent.steps_done += 1
            
            if steps_since_update >= agent.target_update_freq:
                agent.update_target_network()
                steps_since_update = 0
                
            state = next_state
            
            if done:
                break
        
        rewards_history.append(total_reward)
        
        #if episode % 100 == 0:
        avg_reward = np.mean(rewards_history[-100:]) if episode >= 100 else np.mean(rewards_history)
        print(f"DQN - Episódio {episode}/{episodes} - Recompensa: {total_reward:.2f} - Média: {avg_reward:.2f} - Epsilon: {agent.epsilon:.4f}")
    
    env.close()
    return rewards_history

def run_qlearn(episodes=EPISODES, seed=None, max_steps=200):
    """Executa o algoritmo Q-Learning por um número específico de episódios"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    env = gym.make('MountainCar-v0')
    if seed is not None:
        env.reset(seed=seed)
    
    num_bins = (20, 20)
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.1
    
    q_table = np.zeros(num_bins + (env.action_space.n,))
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state, env, num_bins)
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            steps += 1
            
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state, env, num_bins)
            
            q_table[state + (action,)] = (1 - learning_rate) * q_table[state + (action,)] + learning_rate * (
                reward + discount_factor * np.max(q_table[next_state])
            )
            
            state = next_state
            total_reward += reward
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        
        #if episode % 100 == 0:
        avg_reward = np.mean(rewards_history[-100:]) if episode >= 100 else np.mean(rewards_history)
        print(f"Q-Learning - Episódio {episode}/{episodes} - Recompensa: {total_reward:.2f} - Média: {avg_reward:.2f} - Epsilon: {epsilon:.4f}")
    
    env.close()
    return rewards_history, q_table

#========== FUNÇÃO PARA SUAVIZAR CURVAS DE RECOMPENSA ==========
def smooth_rewards(rewards, window=SMOOTHING_WINDOW):
    """Suaviza uma curva de recompensa usando uma média móvel"""
    cumsum = np.cumsum(np.insert(rewards, 0, 0)) 
    return (cumsum[window:] - cumsum[:-window]) / window

#========== EXECUÇÃO PRINCIPAL ==========
def main():
    print("Iniciando experimentos de comparação entre DQN e Q-Learning...")
    
    # Armazenar resultados
    dqn_results = []
    qlearn_results = []
    
    for execution in range(EXECUTIONS):
        execution_seed = SEED + execution
        print(f"\n=== Execução {execution + 1}/{EXECUTIONS} (Seed: {execution_seed}) ===")
        
        # Executar DQN
        print(f"\nIniciando DQN - Execução {execution + 1}...")
        start_time = time.time()
        dqn_rewards = run_dqn(episodes=EPISODES, seed=execution_seed)
        dqn_time = time.time() - start_time
        dqn_results.append(dqn_rewards)
        print(f"DQN concluído em {dqn_time:.2f} segundos")
        
        # Executar Q-Learning
        print(f"\nIniciando Q-Learning - Execução {execution + 1}...")
        start_time = time.time()
        qlearn_rewards, q_table = run_qlearn(episodes=EPISODES, seed=execution_seed, max_steps=2000)
        qlearn_time = time.time() - start_time
        qlearn_results.append(qlearn_rewards)
        print(f"Q-Learning concluído em {qlearn_time:.2f} segundos")
        
        # Salvar Q-table da última execução
        if execution == EXECUTIONS - 1:
            np.save(f"{results_dir}/q_table_mountaincar.npy", q_table)
            
    # Salvar resultados brutos
    np.save(f"{results_dir}/dqn_results.npy", np.array(dqn_results))
    np.save(f"{results_dir}/qlearn_results.npy", np.array(qlearn_results))
            
    # Preparação de dados para visualização
    dqn_df = pd.DataFrame()
    qlearn_df = pd.DataFrame()
    
    # Calcular estatísticas
    for execution in range(EXECUTIONS):
        dqn_rewards = dqn_results[execution]
        qlearn_rewards = qlearn_results[execution]
        
        # Adicionar dados brutos
        dqn_df[f'exec_{execution}'] = dqn_rewards
        qlearn_df[f'exec_{execution}'] = qlearn_rewards
    
    # Calcular médias para cada episódio
    dqn_df['media'] = dqn_df.mean(axis=1)
    qlearn_df['media'] = qlearn_df.mean(axis=1)
    
    # Suavizar resultados
    dqn_smooth = smooth_rewards(dqn_df['media'].values)
    qlearn_smooth = smooth_rewards(qlearn_df['media'].values)
    
    # Criar dataframes com dados suavizados para visualização
    smooth_range = range(SMOOTHING_WINDOW-1, EPISODES)
    viz_data = pd.DataFrame({
        'Episódio': list(smooth_range) * 2,
        'Algoritmo': ['DQN'] * len(smooth_range) + ['Q-Learning'] * len(smooth_range),
        'Recompensa': list(dqn_smooth) + list(qlearn_smooth)
    })
    
    # Vizualização comparativa
    plt.figure(figsize=(16, 10))
    
    # Configurar estilo gráfico
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    # Plotar linhas de recompensa suavizada
    ax = sns.lineplot(
        data=viz_data, 
        x='Episódio', 
        y='Recompensa', 
        hue='Algoritmo',
        palette=['#2E86C1', '#E74C3C'],  # Azul para DQN, Vermelho para Q-Learning
        linewidth=3
    )
    
    # Adicionar áreas sombreadas para desvio padrão
    for i, (df, color) in enumerate(zip([dqn_df, qlearn_df], ['#2E86C1', '#E74C3C'])):
        std = df.iloc[:, :-1].std(axis=1)
        mean = df['media']
        
        # Suavizar desvio padrão
        if len(std) > SMOOTHING_WINDOW:
            std_smooth = smooth_rewards(std)
            mean_smooth = dqn_smooth if i == 0 else qlearn_smooth
            
            plt.fill_between(
                smooth_range,
                mean_smooth - std_smooth,
                mean_smooth + std_smooth,
                alpha=0.2,
                color=color
            )
    
    # Linha horizontal destacando quando o ambiente é considerado resolvido
    plt.axhline(y=-300, color='green', linestyle='--', alpha=0.7, label='Ambiente Resolvido (-300)')
    
    # Adicionar títulos e legendas
    plt.title('Comparação de Desempenho: DQN vs Q-Learning no MountainCar-v0', fontsize=20, pad=20)
    plt.xlabel('Episódio', fontsize=16, labelpad=10)
    plt.ylabel('Recompensa Média (janela=' + str(SMOOTHING_WINDOW) + ')', fontsize=16, labelpad=10)
    
    # Melhorar a legenda
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='lower right', fontsize=14, frameon=True, framealpha=0.9)
    
    # Adicionar anotações informativas
    total_dqn_wins = sum(1 for i in range(len(smooth_range)) if dqn_smooth[i] > qlearn_smooth[i])
    win_percentage = total_dqn_wins / len(smooth_range) * 100
    
    text_x = EPISODES * 0.65
    text_y_start = min(min(dqn_smooth), min(qlearn_smooth)) * 0.8
    
    plt.text(
        text_x, text_y_start,
        f"Estatísticas:\n"
        f"• DQN supera Q-Learning em {win_percentage:.1f}% dos episódios\n"
        f"• Melhor recompensa DQN: {max(dqn_df['media']):.2f}\n"
        f"• Melhor recompensa Q-Learning: {max(qlearn_df['media']):.2f}\n"
        f"• {EXECUTIONS} execuções × {EPISODES} episódios",
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
    )
    
    # Adicionar marca d'água informativa
    plt.annotate(
        f"Comparação gerada em {datetime.now().strftime('%d/%m/%Y')}",
        xy=(0.99, 0.01),
        xycoords='figure fraction',
        fontsize=8,
        color='gray',
        horizontalalignment='right'
    )
    
    # Adicionar grade melhorada
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar layout e salvar
    plt.tight_layout()
    plt.savefig(f"{results_dir}/dqn_vs_qlearn_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{results_dir}/dqn_vs_qlearn_comparison.pdf", format='pdf', bbox_inches='tight')
    
    print(f"\nExperimentos concluídos! Resultados salvos no diretório: {results_dir}")
    print(f"Arquivo de visualização: {results_dir}/dqn_vs_qlearn_comparison.png")
    
    # Exibir o gráfico
    plt.show()
    
    # Carregar e visualizar o desempenho final dos algoritmos
    print("\n=== Resultados Finais ===")
    print(f"DQN - Recompensa média final (últimos 100 episódios): {np.mean(dqn_df['media'].tail(100)):.2f}")
    print(f"Q-Learning - Recompensa média final (últimos 100 episódios): {np.mean(qlearn_df['media'].tail(100)):.2f}")

if __name__ == "__main__":
    main()