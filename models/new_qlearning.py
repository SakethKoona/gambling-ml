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

def plot_win_rate_analysis(test_log, save_dir):
    """
    Plot win rate trends and breakdowns
    """
    episodes = test_log["episodes"]
    
    # Calculate running win rate
    results = []
    for ep in episodes:
        reward = ep["final_reward"]
        if reward > 0:
            results.append(1)  # Win
        elif reward < 0:
            results.append(-1)  # Loss
        else:
            results.append(0)  # Push
    
    # Running win rate calculation
    window = 100
    running_win_rates = []
    running_episodes = []
    
    for i in range(window, len(results)):
        recent_results = results[i-window:i]
        wins = sum(1 for r in recent_results if r == 1)
        total_non_push = sum(1 for r in recent_results if r != 0)
        
        if total_non_push > 0:
            win_rate = wins / total_non_push
        else:
            win_rate = 0
            
        running_win_rates.append(win_rate)
        running_episodes.append(i)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Running Win Rate
    ax1.plot(running_episodes, running_win_rates, color='blue', alpha=0.7)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Win Rate')
    ax1.set_title(f'Running Win Rate ({window}-game window)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Win Rate')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Win/Loss Distribution
    win_count = sum(1 for r in results if r == 1)
    loss_count = sum(1 for r in results if r == -1)
    push_count = sum(1 for r in results if r == 0)
    
    categories = ['Wins', 'Losses', 'Pushes']
    counts = [win_count, loss_count, push_count]
    colors = ['green', 'red', 'gray']
    
    ax2.bar(categories, counts, color=colors, alpha=0.7)
    ax2.set_title('Game Outcome Distribution')
    ax2.set_ylabel('Count')
    
    # Add percentages to bars
    total_games = len(results)
    for i, count in enumerate(counts):
        percentage = (count / total_games) * 100
        ax2.text(i, count + total_games*0.01, f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Win Rate by Player Total (from steps data)
    player_totals_outcomes = {}
    for ep in episodes:
        for step in ep["steps"]:
            total = step["state"]["player_total"]
            reward = step.get("reward", 0)
            
            if total not in player_totals_outcomes:
                player_totals_outcomes[total] = []
            player_totals_outcomes[total].append(reward)
    
    # Calculate win rates by player total
    total_win_rates = {}
    for total, rewards in player_totals_outcomes.items():
        if len(rewards) >= 10:  # Only include totals with sufficient data
            wins = sum(1 for r in rewards if r > 0)
            non_pushes = sum(1 for r in rewards if r != 0)
            if non_pushes > 0:
                total_win_rates[total] = wins / non_pushes
    
    if total_win_rates:
        totals = list(total_win_rates.keys())
        win_rates = list(total_win_rates.values())
        
        ax3.bar(totals, win_rates, alpha=0.7, color='orange')
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax3.set_title('Win Rate by Player Total')
        ax3.set_xlabel('Player Total')
        ax3.set_ylabel('Win Rate')
        ax3.grid(True, alpha=0.3)
    
    # 4. Win Rate by Dealer Upcard
    dealer_outcomes = {}
    for ep in episodes:
        for step in ep["steps"]:
            dealer_card = step["state"]["dealer_upcard"]
            reward = step.get("reward", 0)
            
            if dealer_card not in dealer_outcomes:
                dealer_outcomes[dealer_card] = []
            dealer_outcomes[dealer_card].append(reward)
    
    dealer_win_rates = {}
    for card, rewards in dealer_outcomes.items():
        if len(rewards) >= 10:
            wins = sum(1 for r in rewards if r > 0)
            non_pushes = sum(1 for r in rewards if r != 0)
            if non_pushes > 0:
                dealer_win_rates[card] = wins / non_pushes
    
    if dealer_win_rates:
        cards = list(dealer_win_rates.keys())
        win_rates = list(dealer_win_rates.values())
        
        ax4.bar(cards, win_rates, alpha=0.7, color='purple')
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax4.set_title('Win Rate by Dealer Upcard')
        ax4.set_xlabel('Dealer Upcard')
        ax4.set_ylabel('Win Rate')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/win_rate_analysis.png", dpi=300, bbox_inches='tight')
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

    # 4, 5 & 6. NEW ANALYSES
    analyze_feature_importance(agent, save_dir)
    plot_strategy_heatmap(agent, save_dir)
    plot_win_rate_analysis(test_log, save_dir)  # NEW!

# ----------------------------------------
# 3. Main Training Loop
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--test_episodes", type=int, default=1000)
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
            print(f"Episode {i+1}/{args.episodes} | Epsilon: {agent.epsilon:.4f} | Q-table size: {len(agent.q_table)}")

    print(f"Training complete! Final Q-table size: {len(agent.q_table)}")
    
    # --- TESTING ---
    print(f"\nTesting with {args.test_episodes} episodes...")
    test_log = {
        "episodes": [], 
        "summary": {},
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "algorithm": "qlearning",
            "training_episodes": args.episodes,
            "test_episodes": args.test_episodes,
            "learning_rate": agent.lr,
            "discount_factor": agent.gamma,
            "final_epsilon": agent.epsilon,
            "q_table_size": len(agent.q_table)
        }
    }
    
    total_reward = 0
    
    for i in range(args.test_episodes):
        state = env.reset()
        done = False
        episode_data = {
            "episode": i,
            "steps": [], 
            "final_reward": 0,
            "player_hands": [],
            "dealer_cards": [],
            "total_bets": 0
        }
        
        while not done:
            # train=False means purely Greedy
            action = agent.choose_action(state, train=False)
            
            step_info = {
                "state": {
                    "player_total": state[0],
                    "is_soft": bool(state[1]),
                    "dealer_upcard": state[2],
                    "can_double": bool(state[3]),
                    "can_split": bool(state[4])
                },
                "action": int(action),
                "action_name": Action(action).name
            }
            episode_data["steps"].append(step_info)
            
            next_state, reward, done, _ = env.step(Action(action))
            
            # Log the reward back into step_info
            step_info["reward"] = reward 

            state = next_state
            
            if done:
                episode_data["final_reward"] = reward
                episode_data["player_hands"] = env.player_hands.copy()
                episode_data["dealer_cards"] = env.dealer_cards.copy()
                episode_data["total_bets"] = sum(env.bets)
                total_reward += reward

        test_log["episodes"].append(episode_data)

    # Calculate comprehensive summary statistics
    test_log["summary"] = {
        "total_reward": total_reward,
        "average_reward": total_reward / args.test_episodes,
        "total_winnings": env.total_winnings,
        "total_bets": env.total_bets,
        "rounds_played": env.rounds_played,
        "rounds_won": env.rounds_won,
        "rounds_lost": env.rounds_lost,
        "rounds_push": env.rounds_push,
        "win_rate": env.rounds_won / env.rounds_played if env.rounds_played > 0 else 0,
        "roi": env.total_winnings / env.total_bets if env.total_bets > 0 else 0
    }

    # --- SAVING RESULTS ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"plots/q_learning_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Save JSON log
    json_filename = f"logs/test_results_qlearning_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(test_log, f, indent=2)
    
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
                "Can_Double": step["state"]["can_double"],
                "Can_Split": step["state"]["can_split"],
                "Action_Taken": step["action_name"],
                "Step_Reward": step.get("reward", 0),
                "Game_Result": "In Progress"
            }
            csv_rows.append(row)
            step_count += 1
        if csv_rows:
            csv_rows[-1]["Game_Result"] = "WON" if ep_data["final_reward"] > 0 else ("LOST" if ep_data["final_reward"] < 0 else "PUSH")

    df = pd.DataFrame(csv_rows)
    df.to_csv(f"logs/q_learning_{timestamp}.csv", index=False)

    # Generate all plots
    plot_results(test_log, save_dir, agent)
    
    # Print comprehensive results
    summary = test_log["summary"]
    print(f"\n" + "="*60)
    print("Q-LEARNING RESULTS")
    print("="*60)
    print(f"Training episodes:     {args.episodes:,}")
    print(f"Test episodes:         {args.test_episodes:,}")
    print(f"Q-table size:          {len(agent.q_table):,} states")
    print(f"Final epsilon:         {agent.epsilon:.4f}")
    print(f"\nPerformance Metrics:")
    print(f"Total reward:          {summary['total_reward']}")
    print(f"Average reward:        {summary['average_reward']:.3f}")
    print(f"Win rate:              {summary['win_rate']:.3f} ({summary['rounds_won']}/{summary['rounds_played']})")
    print(f"ROI:                   {summary['roi']:.3f}")
    print(f"Rounds won:            {summary['rounds_won']}")
    print(f"Rounds lost:           {summary['rounds_lost']}")
    print(f"Rounds pushed:         {summary['rounds_push']}")
    print(f"\nFiles saved:")
    print(f"JSON log:              {json_filename}")
    print(f"CSV data:              logs/q_learning_{timestamp}.csv")
    print(f"Plots:                 {save_dir}/")