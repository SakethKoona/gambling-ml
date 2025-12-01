import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import random

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.blackjack import BlackjackEnv
from envs.utils import Action

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ----------------------------------------
# 1. The Q-Learning Agent
# ----------------------------------------
class QLearningAgent:
    def __init__(self, env, learning_rate=0.01, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_actions = 4 # HIT, STAND, DOUBLE, SPLIT
        self.q_table = {}

    def get_state_key(self, state):
        # state: (player_total, is_soft, dealer_upcard, can_double, can_split)
        return tuple(state)

    def choose_action(self, state, train=True):
        # If we are training, use Epsilon-Greedy (Exploration)
        if train and random.random() < self.epsilon:
            return random.choice(range(self.n_actions))
        
        # If testing (or greedy choice), pick the best known move
        key = self.get_state_key(state)
        if key not in self.q_table:
            return random.choice(range(self.n_actions)) # If unknown, guess
        
        # Pick action with highest Q-value
        return int(np.argmax(self.q_table[key]))

    def learn(self, state, action, reward, next_state, done):
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)

        # Calculate Target
        if done:
            target = reward
        else:
            next_key = self.get_state_key(next_state)
            if next_key not in self.q_table:
                self.q_table[next_key] = np.zeros(self.n_actions)
            target = reward + self.gamma * np.max(self.q_table[next_key])

        # Update Q-Value
        self.q_table[key][action] += self.lr * (target - self.q_table[key][action])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# ----------------------------------------
# 2. Advanced Analysis & Plotting
# ----------------------------------------

def analyze_feature_importance(agent, save_dir):
    """
    Calculates how much the 'Value' (Best Q-value) varies for each feature.
    High Variance = High Importance (The feature drastically changes expected reward).
    """
    # Convert Q-Table to list of dicts
    data = []
    for state_tuple, q_values in agent.q_table.items():
        best_value = np.max(q_values)
        data.append({
            'Player Total': state_tuple[0],
            'Is Soft Hand': "Yes" if state_tuple[1] else "No",
            'Dealer Upcard': state_tuple[2],
            'Can Double': "Yes" if state_tuple[3] else "No",
            'Can Split': "Yes" if state_tuple[4] else "No",
            'Best_Value': best_value
        })
    
    if not data:
        print("Warning: Q-Table empty or not populated enough for feature analysis.")
        return

    df = pd.DataFrame(data)
    
    # Calculate Standard Deviation of Value for each feature
    importance = {}
    features = ['Player Total', 'Is Soft Hand', 'Dealer Upcard', 'Can Double', 'Can Split']
    
    for f in features:
        # Group by feature and see how much the Result changes
        if f in df.columns:
            variance = df.groupby(f)['Best_Value'].mean().std()
            importance[f] = variance if not np.isnan(variance) else 0

    # Normalize
    total_var = sum(importance.values())
    if total_var == 0:
        total_var = 1 # Prevent divide by zero
        
    for k in importance:
        importance[k] = (importance[k] / total_var) * 100

    # Plot
    plt.figure(figsize=(10, 6))
    imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance (%)'])
    imp_df = imp_df.sort_values(by='Importance (%)', ascending=False)
    
    sns.barplot(x='Importance (%)', y='Feature', data=imp_df, palette='viridis')
    plt.title('Q-Learning Feature Importance\n(Impact on Win Rate)')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importance.png")
    plt.close()

def plot_strategy_heatmap(agent, save_dir):
    """
    Generates a traditional 'Blackjack Strategy Card' heatmap.
    """
    player_totals = range(12, 22)
    dealer_cards = range(2, 12) # 11 is Ace
    
    heatmap_data = np.zeros((len(player_totals), len(dealer_cards)))
    
    for i, pt in enumerate(player_totals):
        for j, dc in enumerate(dealer_cards):
            # Assume hard hand, no special moves
            state = (pt, 0, dc, 0, 0) 
            key = agent.get_state_key(state)
            if key in agent.q_table:
                action = int(np.argmax(agent.q_table[key]))
                heatmap_data[i, j] = action
            else:
                heatmap_data[i, j] = 0 # Default Stand

    plt.figure(figsize=(10, 8))
    cmap = sns.color_palette("Set3", 4) 
    sns.heatmap(heatmap_data, cmap=cmap, annot=True, cbar=False,
                xticklabels=dealer_cards, yticklabels=player_totals)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap[0], label='Stand'),
        Patch(facecolor=cmap[1], label='Hit'),
        Patch(facecolor=cmap[2], label='Double'),
        Patch(facecolor=cmap[3], label='Split')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title("Learned Policy: Hard Hands (12-21)")
    plt.xlabel("Dealer Upcard")
    plt.ylabel("Player Total")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/strategy_heatmap.png")
    plt.close()

def plot_results(test_log, save_dir, agent):
    # 1. Cumulative Reward
    episodes = test_log["episodes"]
    rewards = [ep["final_reward"] for ep in episodes]
    cumulative = np.cumsum(rewards)
    
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative, color='darkblue', linewidth=2)
    plt.title(f"Q-Learning Cumulative Reward (ROI: {test_log['summary']['roi']:.3f})")
    plt.xlabel("Episode")
    plt.ylabel("Total Winnings")
    plt.axhline(0, color='red', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/cumulative_reward.png")
    plt.close()

    # 2. Action Distribution
    all_actions = []
    for ep in episodes:
        for step in ep["steps"]:
            all_actions.append(step["action_name"])
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x=all_actions, order=['HIT', 'STAND', 'DOUBLE_DOWN', 'SPLIT'])
    plt.title("Action Distribution (Q-Learning)")
    plt.savefig(f"{save_dir}/action_dist.png")
    plt.close()

    # 3. Strategy Heatmap (Soft Hands)
    soft_data = []
    for ep in episodes:
        for step in ep["steps"]:
            if step["state"]["is_soft"]:
                soft_data.append({
                    "Total": step["state"]["player_total"],
                    "Action": step["action_name"]
                })
    
    if soft_data:
        df = pd.DataFrame(soft_data)
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x="Total", hue="Action", multiple="stack", binwidth=1)
        plt.title("Strategy on SOFT Hands")
        plt.savefig(f"{save_dir}/soft_hand_strategy.png")
        plt.close()

    # 4 & 5. NEW ANALYSES
    analyze_feature_importance(agent, save_dir)
    plot_strategy_heatmap(agent, save_dir)

# ----------------------------------------
# 3. Main Training Loop
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000)
    args = parser.parse_args()

    env = BlackjackEnv(n_decks=4)
    agent = QLearningAgent(env)

    # --- TRAINING ---
    print(f"Training Q-Learning Agent for {args.episodes} episodes...")
    for i in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, train=True)
            next_state, reward, done, _ = env.step(Action(action))
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        
        agent.decay_epsilon()
        
        if (i+1) % 10000 == 0:
            print(f"Episode {i+1}/{args.episodes} | Epsilon: {agent.epsilon:.4f}")

    # --- TESTING ---
    print("\nTraining complete. Running 1,000 Test Games...")
    test_log = {"episodes": [], "summary": {}}
    total_reward = 0
    
    for i in range(20000):
        state = env.reset()
        done = False
        episode_data = {"steps": [], "final_reward": 0}
        
        while not done:
            # train=False means purely Greedy
            action = agent.choose_action(state, train=False)
            
            step_info = {
                "state": {
                    "player_total": state[0],
                    "is_soft": bool(state[1]),
                    "dealer_upcard": state[2],
                    "can_split": bool(state[4])
                },
                "action_name": Action(action).name
            }
            episode_data["steps"].append(step_info)
            
            next_state, reward, done, _ = env.step(Action(action))
            
            # --- FIX: LOG THE REWARD BACK INTO STEP_INFO ---
            step_info["reward"] = reward 
            # -----------------------------------------------

            state = next_state
            
            if done:
                episode_data["final_reward"] = reward
                total_reward += reward

        test_log["episodes"].append(episode_data)

    # --- SAVING RESULTS ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"plots/q_learning_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    test_log["summary"]["roi"] = total_reward / env.total_bets
    test_log["summary"]["total_reward"] = total_reward
    
    # Save CSV
    csv_rows = []
    for episode_idx, ep_data in enumerate(test_log["episodes"]):
        step_count = 0
        for step in ep_data["steps"]:
            row = {
                "Episode_ID": episode_idx + 1,
                "Step_Number": step_count + 1,
                "Player_Total": step["state"]["player_total"],
                "Dealer_Upcard": step["state"]["dealer_upcard"],
                "Is_Soft_Hand": step["state"]["is_soft"],
                "Action_Taken": step["action_name"],
                "Step_Reward": step.get("reward", 0), # Now this will find the reward!
                "Game_Result": "In Progress"
            }
            csv_rows.append(row)
            step_count += 1
        if csv_rows:
            csv_rows[-1]["Game_Result"] = "WON" if ep_data["final_reward"] > 0 else ("LOST" if ep_data["final_reward"] < 0 else "PUSH")

    df = pd.DataFrame(csv_rows)
    df.to_csv(f"logs/q_learning_{timestamp}.csv", index=False)

    plot_results(test_log, save_dir, agent)
    
    print(f"\nResults saved to {save_dir}")
    print(f"Total Reward: {total_reward}")
    print(f"ROI: {test_log['summary']['roi']:.4f}")