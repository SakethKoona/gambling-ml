import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from envs.blackjack import BlackjackEnv, generate_dataset, stochastic_policy, simple_policy
from envs.utils import Action
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_rewards_over_time(test_log, save_path=None):
    """Plot cumulative reward and per-episode reward over time"""
    episodes = test_log["episodes"]
    episode_numbers = [ep["episode"] for ep in episodes]
    episode_rewards = [ep["final_reward"] for ep in episodes]
    cumulative_rewards = np.cumsum(episode_rewards)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot cumulative reward
    ax1.plot(episode_numbers, cumulative_rewards, linewidth=2, color='darkblue', alpha=0.8)
    ax1.set_title('Cumulative Reward Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot per-episode reward with smoothing
    ax2.bar(episode_numbers, episode_rewards, alpha=0.6, color='lightcoral', label='Episode Reward')
    
    # Add rolling average
    window = min(50, len(episode_rewards) // 10)
    if window > 1:
        rolling_avg = pd.Series(episode_rewards).rolling(window=window, center=True).mean()
        ax2.plot(episode_numbers, rolling_avg, color='darkred', linewidth=2, 
                label=f'{window}-Episode Rolling Average')
    
    ax2.set_title('Per-Episode Reward Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Reward')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_rewards.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_action_distribution(test_log, save_path=None):
    """Plot distribution of actions taken"""
    # Collect all actions from all episodes
    all_actions = []
    for episode in test_log["episodes"]:
        for step in episode["steps"]:
            all_actions.append(step["action_name"])
    
    action_counts = Counter(all_actions)
    
    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(actions)))
    
    bars = ax.bar(actions, counts, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}\n({count/sum(counts)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Action Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Action')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_actions.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_action_distribution_by_state(test_log, save_path=None):
    """Plot action distribution broken down by game state"""
    # Collect actions with state information
    state_actions = []
    for episode in test_log["episodes"]:
        for step in episode["steps"]:
            state_actions.append({
                'player_total': step["state"]["player_total"],
                'dealer_upcard': step["state"]["dealer_upcard"],
                'action': step["action_name"],
                'is_soft': step["state"]["is_soft"],
                'can_double': step["state"]["can_double"],
                'can_split': step["state"]["can_split"]
            })
    
    df = pd.DataFrame(state_actions)
    
    # Plot 1: Actions by player total
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Actions by player total
    action_by_total = df.groupby(['player_total', 'action']).size().unstack(fill_value=0)
    action_by_total_pct = action_by_total.div(action_by_total.sum(axis=1), axis=0) * 100
    
    action_by_total_pct.plot(kind='bar', stacked=True, ax=axes[0,0], colormap='Set3')
    axes[0,0].set_title('Action Distribution by Player Total', fontweight='bold')
    axes[0,0].set_xlabel('Player Total')
    axes[0,0].set_ylabel('Percentage')
    axes[0,0].legend(title='Action', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Actions by dealer upcard
    action_by_dealer = df.groupby(['dealer_upcard', 'action']).size().unstack(fill_value=0)
    action_by_dealer_pct = action_by_dealer.div(action_by_dealer.sum(axis=1), axis=0) * 100
    
    action_by_dealer_pct.plot(kind='bar', stacked=True, ax=axes[0,1], colormap='Set3')
    axes[0,1].set_title('Action Distribution by Dealer Upcard', fontweight='bold')
    axes[0,1].set_xlabel('Dealer Upcard')
    axes[0,1].set_ylabel('Percentage')
    axes[0,1].legend(title='Action', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].tick_params(axis='x', rotation=0)
    
    # Actions for soft vs hard hands
    action_by_soft = df.groupby(['is_soft', 'action']).size().unstack(fill_value=0)
    action_by_soft_pct = action_by_soft.div(action_by_soft.sum(axis=1), axis=0) * 100
    action_by_soft_pct.index = ['Hard', 'Soft']
    
    action_by_soft_pct.plot(kind='bar', stacked=True, ax=axes[1,0], colormap='Set3')
    axes[1,0].set_title('Action Distribution: Soft vs Hard Hands', fontweight='bold')
    axes[1,0].set_xlabel('Hand Type')
    axes[1,0].set_ylabel('Percentage')
    axes[1,0].legend(title='Action', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,0].tick_params(axis='x', rotation=0)
    
    # Special actions availability
    special_actions = df[(df['can_double'] == True) | (df['can_split'] == True)]
    if not special_actions.empty:
        special_counts = special_actions['action'].value_counts()
        special_counts.plot(kind='bar', ax=axes[1,1], color='lightblue', alpha=0.8)
        axes[1,1].set_title('Actions When Special Moves Available', fontweight='bold')
        axes[1,1].set_xlabel('Action')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].tick_params(axis='x', rotation=45)
    else:
        axes[1,1].text(0.5, 0.5, 'No special actions\navailable in dataset', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Actions When Special Moves Available', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_detailed_actions.png", dpi=300, bbox_inches='tight')
    plt.show()

def train_supervised_model(model_type='logistic', n_samples=100000, policy=simple_policy):
    # Generate dataset
    data = generate_dataset(n_samples, policy=policy)
    X = data[['player_total', 'is_soft', 'dealer_upcard', 'can_double', 'can_split']]
    y = data['action']

    # Initialize model
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("Unsupported model type")

    # Train model
    model.fit(X, y)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='logistic', help='Type of model to train: logistic or random_forest')
    parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples to generate for training')
    parser.add_argument('--n_decks', type=int, default=4, help='Number of decks in the Blackjack environment')
    parser.add_argument('--policy', type=str, default='simple', help='Policy to generate training data: simple or stochastic')
    parser.add_argument('--no_plots', action='store_true', help='Skip plotting')
    args = parser.parse_args()

    policy_map = {
        'simple': simple_policy,
        'stochastic': stochastic_policy
    }

    if args.policy not in policy_map:
        raise ValueError("Unsupported policy type. Choose 'simple' or 'stochastic'.")

    model = train_supervised_model(model_type=args.model_type, n_samples=args.n_samples, policy=policy_map[args.policy])
    print(f"Trained {args.model_type} model on {args.n_samples} samples.")

    # Next, run the model through several test episodes to evaluate performance
    env = BlackjackEnv(n_decks=args.n_decks)
    n_test_episodes = 1000
    total_reward = 0
    
    # Initialize logging data
    test_log = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_type": args.model_type,
            "training_samples": args.n_samples,
            "training_policy": args.policy,
            "n_decks": args.n_decks,
            "n_test_episodes": n_test_episodes
        },
        "episodes": []
    }
    
    for ep in range(n_test_episodes):
        state = env.reset()
        done = False
        episode_log = {
            "episode": ep,
            "steps": [],
            "final_reward": 0,
            "player_hands": [],
            "dealer_cards": [],
            "total_bets": 0
        }
        
        step_count = 0
        while not done:
            state_array = np.array(state).reshape(1, -1)
            action = model.predict(state_array)[0]
            
            # Log the step
            step_log = {
                "step": step_count,
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
            
            next_state, reward, done, info = env.step(action)
            
            step_log["reward"] = reward
            step_log["done"] = done
            episode_log["steps"].append(step_log)
            
            state = next_state
            step_count += 1
        
        # Record episode summary
        episode_log["final_reward"] = reward
        episode_log["player_hands"] = env.player_hands.copy()
        episode_log["dealer_cards"] = env.dealer_cards.copy()
        episode_log["total_bets"] = sum(env.bets)
        
        test_log["episodes"].append(episode_log)
        total_reward += reward

    # Add summary statistics
    test_log["summary"] = {
        "total_reward": total_reward,
        "average_reward": total_reward / n_test_episodes,
        "total_winnings": env.total_winnings,
        "total_bets": env.total_bets,
        "rounds_played": env.rounds_played,
        "rounds_won": env.rounds_won,
        "rounds_lost": env.rounds_lost,
        "rounds_push": env.rounds_push,
        "win_rate": env.rounds_won / env.rounds_played if env.rounds_played > 0 else 0,
        "roi": env.total_winnings / env.total_bets if env.total_bets > 0 else 0
    }

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory structure
    logs_dir = "logs"
    plots_base_dir = "plots"
    plots_run_dir = f"{plots_base_dir}/{args.model_type}_{args.policy}_{timestamp}"
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_run_dir, exist_ok=True)
    
    # File paths
    json_filename = f"{logs_dir}/test_results_{args.model_type}_{args.policy}_{timestamp}.json"
    plot_save_path = f"{plots_run_dir}/plot"  # Base path for plots in the timestamped folder
    
    # Save JSON log
    with open(json_filename, 'w') as f:
        json.dump(test_log, f, indent=2)

    print(f"Average reward over {n_test_episodes} test episodes: {total_reward / n_test_episodes}")
    print(f"Total reward: {total_reward}")
    print(f"Test results saved to: {json_filename}")
    print(f"Win rate: {test_log['summary']['win_rate']:.3f}")
    print(f"ROI: {test_log['summary']['roi']:.3f}")

    # Generate plots unless disabled
    if not args.no_plots:
        print("Generating plots...")
        plot_rewards_over_time(test_log, plot_save_path)
        plot_action_distribution(test_log, plot_save_path)
        plot_action_distribution_by_state(test_log, plot_save_path)
        print(f"Plots saved in directory: {plots_run_dir}")
        print(f"  - {plot_save_path}_rewards.png")
        print(f"  - {plot_save_path}_actions.png") 
        print(f"  - {plot_save_path}_detailed_actions.png")



