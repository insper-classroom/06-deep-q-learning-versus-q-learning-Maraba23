import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Aumentado para 128 unidades
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
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_dec=0.995, 
                 episodes=1000, batch_size=64, memory_size=100000, max_steps=200, 
                 lr=0.001, target_update_freq=100, device="cpu"):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.max_steps = max_steps  # 200 é o padrão do MountainCar
        self.device = device
        self.target_update_freq = target_update_freq  # Atualiza a cada 100 steps, não episódios
        self.steps_done = 0

        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        self.model = DQN(input_dim, output_dim).to(self.device)
        self.target_model = DQN(input_dim, output_dim).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print("Target network updated!")

    def select_action(self, state):
        # Normaliza o estado antes de usar
        state = normalize_state(state)
        
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def experience(self, state, action, reward, next_state, done):
        # Normaliza os estados antes de armazenar
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

        # Double DQN: seleciona ações usando a rede principal
        current_q_values = self.model(states).gather(1, actions).squeeze()
        
        # Mas avalia essas ações usando a rede alvo
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze()
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Calcula a perda e atualiza a rede
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidade
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Decrementa epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec
            
        return loss.item()

def train_agent():
    # Configuração do ambiente
    env = gym.make('MountainCar-v0')
    seed = 42
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Detecção de GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Inicializa o agente com os parâmetros melhorados
    agent = DeepQLearning(
        env=env, 
        device=device,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_dec=0.995,  # Decaimento mais rápido
        episodes=500,       # Menos episódios são necessários com melhor learning
        batch_size=64,
        memory_size=100000,
        max_steps=200,      # Define para o padrão do ambiente
        lr=0.001,           # Taxa de aprendizado maior
        target_update_freq=100  # Atualiza a cada 100 passos
    )

    rewards_history = []
    best_reward = -float('inf')
    steps_since_update = 0
    solved = False

    print("Iniciando treinamento...")
    for episode in range(agent.episodes):
        state, _ = env.reset()
        total_reward = 0
        original_total_reward = 0
        total_loss = 0
        
        for step in range(agent.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Reward shaping - modifica a recompensa para aprendizado mais rápido
            shaped_reward = shape_reward(state, reward, done)
            
            # Armazena a experiência com a recompensa modificada
            agent.experience(state, action, shaped_reward, next_state, done)
            
            # Rastreia ambas as recompensas
            total_reward += shaped_reward
            original_total_reward += reward
            
            # Atualiza o agente várias vezes para aproveitar melhor os dados
            for _ in range(4):  # 4 atualizações por passo
                loss = agent.experience_replay()
                if loss > 0:
                    total_loss += loss
            
            steps_since_update += 1
            agent.steps_done += 1
            
            # Atualiza a rede alvo periodicamente
            if steps_since_update >= agent.target_update_freq:
                agent.update_target_network()
                steps_since_update = 0
                
            state = next_state
            
            if done:
                break
        
        # Registra o desempenho
        rewards_history.append(original_total_reward)
        avg_loss = total_loss / (step + 1) if step > 0 else 0
        
        # Mostra progresso periodicamente
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:]) if episode >= 10 else np.mean(rewards_history)
            print(f"Episódio {episode + 1}/{agent.episodes} - Recompensa original: {original_total_reward:.2f} - "
                  f"Recompensa modificada: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f} - "
                  f"Média últimos 10: {avg_reward:.2f} - Loss: {avg_loss:.4f}")
        
        # Verifica se resolveu o ambiente (atingiu a bandeira)
        if original_total_reward > -100:  # Se completou em menos de 100 passos
            if not solved:
                print(f"\nAmbiente resolvido no episódio {episode + 1}!")
                solved = True
            
            # Guarda o melhor modelo
            if original_total_reward > best_reward:
                best_reward = original_total_reward
                torch.save(agent.model.state_dict(), "mountaincar_best_model.pth")
                print(f"Novo melhor modelo salvo! Recompensa: {best_reward:.2f}")

    env.close()
    print("\nTreinamento concluído!")
    
    # Plota o resultado
    plt.figure(figsize=(12, 6))
    
    # Suaviza a curva de recompensa para melhor visualização
    window_size = 20
    smoothed_rewards = []
    for i in range(len(rewards_history)):
        start_idx = max(0, i - window_size)
        smoothed_rewards.append(np.mean(rewards_history[start_idx:i+1]))
    
    plt.plot(rewards_history, alpha=0.3, color='blue', label="Recompensa por episódio")
    plt.plot(smoothed_rewards, linewidth=2, color='darkblue', label=f"Média móvel (janela={window_size})")
    
    plt.axhline(y=-110, color='r', linestyle='--', alpha=0.5, label="Ambiente resolvido")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Original")
    plt.title("Double DQN com Reward Shaping - MountainCar-v0")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("double_dqn_mountaincar_improved.png")
    plt.show()

    print("Gráfico salvo como double_dqn_mountaincar_improved.png")
    return agent

# Executa o treinamento
if __name__ == "__main__":
    agent = train_agent()