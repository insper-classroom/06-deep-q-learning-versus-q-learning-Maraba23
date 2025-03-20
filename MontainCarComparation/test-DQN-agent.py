import gymnasium as gym
import numpy as np
import torch
import time
from matplotlib import pyplot as plt

# Importa classes e funções necessárias do arquivo de treinamento
# Certifique-se de ter o arquivo do modelo de treinamento no mesmo diretório
from DeepQLearn import DQN, normalize_state

def test_agent(model_path="mountaincar_best_model.pth", num_episodes=5, render_mode="human"):
    """
    Testa um agente DQN treinado no ambiente MountainCar com visualização.
    
    Args:
        model_path: Caminho para o modelo treinado
        num_episodes: Número de episódios para testar
        render_mode: Modo de renderização ('human' para visualização na tela)
    """
    # Configura o ambiente com renderização
    env = gym.make('MountainCar-v0', render_mode=render_mode)
    
    # Carrega o modelo treinado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modelo carregado com sucesso de: {model_path}")
    except FileNotFoundError:
        print(f"Erro: Arquivo de modelo {model_path} não encontrado!")
        print("Verifique se você executou o treinamento completo anteriormente.")
        return
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return
    
    # Coloca o modelo em modo de avaliação
    model.eval()
    
    rewards = []
    steps_list = []
    
    print("\n===== Iniciando teste do agente =====")
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        print(f"\nEpisódio {episode+1}/{num_episodes}")
        print("Estado inicial:", state)
        
        # Loop do episódio
        while not done:
            # Normaliza o estado
            norm_state = normalize_state(state)
            
            # Seleciona a ação com base no modelo (sem exploração aleatória)
            state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
            
            # Executa a ação no ambiente
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Atualiza o estado e métricas
            state = next_state
            total_reward += reward
            steps += 1
            
            # Pequena pausa para melhor visualização (opcional)
            if render_mode == "human":
                time.sleep(0.01)
                
            # Mostra informações sobre o estado atual (opcional)
            if steps % 20 == 0:
                print(f"  Passo {steps}: Posição = {state[0]:.3f}, Velocidade = {state[1]:.3f}")
        
        # Registra resultados do episódio
        rewards.append(total_reward)
        steps_list.append(steps)
        
        # Mostra resultado do episódio
        print(f"Episódio {episode+1} concluído:")
        print(f"  Passos: {steps}")
        print(f"  Recompensa total: {total_reward}")
        print(f"  Posição final: {state[0]:.3f}")
        
        # Se conseguiu resolver, celebre!
        if state[0] >= 0.5:
            print("  🎉 Sucesso! O carro alcançou o topo da montanha!")
        else:
            print("  ❌ Falha: O carro não alcançou o topo da montanha.")
    
    env.close()
    
    # Exibe estatísticas gerais
    print("\n===== Resumo do teste =====")
    print(f"Recompensa média: {np.mean(rewards):.2f}")
    print(f"Média de passos: {np.mean(steps_list):.2f}")
    print(f"Taxa de sucesso: {sum([1 for r in rewards if r > -200]) / num_episodes * 100:.1f}%")
    
    # Plota os resultados
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].bar(range(1, num_episodes+1), rewards)
    ax[0].set_xlabel("Episódio")
    ax[0].set_ylabel("Recompensa")
    ax[0].set_title("Recompensas por Episódio")
    ax[0].grid(alpha=0.3)
    
    ax[1].bar(range(1, num_episodes+1), steps_list)
    ax[1].set_xlabel("Episódio")
    ax[1].set_ylabel("Passos")
    ax[1].set_title("Passos por Episódio")
    ax[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("test_results.png")
    plt.show()

if __name__ == "__main__":
    # Você pode alterar o nome do arquivo do modelo se necessário
    test_agent(model_path="mountaincar_best_model.pth", num_episodes=5, render_mode="human")