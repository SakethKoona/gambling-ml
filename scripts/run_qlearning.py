import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import json
from datetime import datetime
from envs.blackjack import BlackjackEnv
from envs.utils import Action
from models.qlearning import QLearningAgent
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
    ax1.set_title('Q-Learning: Cumulative Reward Over Time', fontsize=14, fontweight='bold')
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
    
    ax2.set_title('Q-Learning: Per-Episode Reward Over Time', fontsize=14, fontweight='bold')
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
    
    ax.set_title('Q-Learning: Action Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Action')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_actions.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_progress(training_log, save_path=None):
    """Plot training progress metrics"""
    episodes = training_log["episodes"]
    episode_numbers = [ep["episode"] for ep in episodes]
    episode_rewards = [ep["reward"] for ep in episodes]
    epsilon_values = [ep["epsilon"] for ep in episodes]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot training rewards
    window = min(100, len(episode_rewards) // 20)
    if window > 1:
        rolling_avg = pd.Series(episode_rewards).rolling(window=window, center=True).mean()
        ax1.plot(episode_numbers, rolling_avg, color='green', linewidth=2, 
                label=f'{window}-Episode Rolling Average')
    
    ax1.scatter(episode_numbers[::50], [episode_rewards[i] for i in range(0, len(episode_rewards), 50)], 
               alpha=0.3, s=1, color='lightgreen', label='Episode Rewards (sampled)')
    ax1.set_title('Q-Learning Training: Episode Rewards', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot epsilon decay
    ax2.plot(episode_numbers, epsilon_values, color='orange', linewidth=2)
    ax2.set_title('Q-Learning Training: Epsilon Decay', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Episode')
    ax2.set_ylabel('Epsilon (Exploration Rate)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_training.png", dpi=300, bbox_inches='tight')
    plt.show()

def run_qlearning_experiment(training_episodes=50000, test_episodes=1000, 
                           learning_rate=0.1, discount_factor=0.99, 
                           epsilon=0.1, n_decks=4):
    """Run complete Q-learning experiment with training and testing"""
    
    print(f"Starting Q-learning experiment...")
    print(f"Training episodes: {training_episodes}")
    print(f"Test episodes: {test_episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Discount factor: {discount_factor}")
    print(f"Initial epsilon: {epsilon}")
    print("=" * 50)
    
    # Initialize environment and agent
    env = BlackjackEnv(n_decks=n_decks)
    agent = QLearningAgent(env, learning_rate=learning_rate, 
                          discount_factor=discount_factor, epsilon=epsilon)
    
    # Training phase with progress tracking
    print("Training Q-learning agent...")
    training_log = {"episodes": []}
    
    for ep in range(training_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(Action(action))
            agent.learn(state, action, reward, next_state, done)
            
            if done:
                episode_reward = reward
            
            state = next_state
        
        # Log training progress (sample every 100 episodes to avoid memory issues)
        if ep % 100 == 0:
            training_log["episodes"].append({
                "episode": ep,
                "reward": episode_reward,
                "epsilon": agent.epsilon,
                "q_table_size": len(agent.q_table)
            })
        
        # Decay epsilon
        if ep % 1000 == 0 and agent.epsilon > 0.01:
            agent.epsilon *= 0.995
        
        if ep % 5000 == 0:
            print(f"Training episode {ep}, epsilon: {agent.epsilon:.3f}, Q-table size: {len(agent.q_table)}")
    
    print(f"Training complete! Final Q-table size: {len(agent.q_table)}")
    
    # Testing phase
    print(f"\nTesting trained agent over {test_episodes} episodes...")
    
    # Set epsilon to 0 for testing (pure exploitation)
    test_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    test_log = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "algorithm": "qlearning",
            "training_episodes": training_episodes,
            "test_episodes": test_episodes,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "initial_epsilon": epsilon,
            "final_epsilon": test_epsilon,
            "n_decks": n_decks,
            "q_table_size": len(agent.q_table)
        },
        "episodes": []
    }
    
    total_reward = 0
    
    for ep in range(test_episodes):
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
            action = agent.choose_action(state)
            
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
                "action_name": Action(action).name,
                "q_values": agent.q_table.get(agent.get_state_key(state), np.zeros(4)).tolist()
            }
            
            next_state, reward, done, info = env.step(Action(action))
            
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
        "average_reward": total_reward / test_episodes,
        "total_winnings": env.total_winnings,
        "total_bets": env.total_bets,
        "rounds_played": env.rounds_played,
        "rounds_won": env.rounds_won,
        "rounds_lost": env.rounds_lost,
        "rounds_push": env.rounds_push,
        "win_rate": env.rounds_won / env.rounds_played if env.rounds_played > 0 else 0,
        "roi": env.total_winnings / env.total_bets if env.total_bets > 0 else 0
    }
    
    # Restore epsilon
    agent.epsilon = test_epsilon
    
    return test_log, training_log, agent

def main():
    parser = argparse.ArgumentParser(description="Q-Learning Blackjack Experiment")
    parser.add_argument('--training_episodes', type=int, default=50000, 
                       help='Number of training episodes')
    parser.add_argument('--test_episodes', type=int, default=1000, 
                       help='Number of test episodes')
    parser.add_argument('--learning_rate', type=float, default=0.1, 
                       help='Learning rate (alpha)')
    parser.add_argument('--discount_factor', type=float, default=0.99, 
                       help='Discount factor (gamma)')
    parser.add_argument('--epsilon', type=float, default=0.1, 
                       help='Initial epsilon for epsilon-greedy')
    parser.add_argument('--n_decks', type=int, default=4, 
                       help='Number of decks in blackjack')
    parser.add_argument('--no_plots', action='store_true', 
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Run experiment
    test_log, training_log, agent = run_qlearning_experiment(
        training_episodes=args.training_episodes,
        test_episodes=args.test_episodes,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        n_decks=args.n_decks
    )
    
    # Generate timestamp and create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = "logs"
    plots_base_dir = "plots"
    plots_run_dir = f"{plots_base_dir}/qlearning_{timestamp}"
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_run_dir, exist_ok=True)
    
    # Save results
    test_filename = f"{logs_dir}/test_results_qlearning_{timestamp}.json"
    training_filename = f"{logs_dir}/training_log_qlearning_{timestamp}.json"
    plot_save_path = f"{plots_run_dir}/plot"
    
    with open(test_filename, 'w') as f:
        json.dump(test_log, f, indent=2)
    
    with open(training_filename, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Print results
    summary = test_log["summary"]
    print(f"\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f"Average reward over {args.test_episodes} test episodes: {summary['average_reward']:.3f}")
    print(f"Total reward: {summary['total_reward']}")
    print(f"Win rate: {summary['win_rate']:.3f}")
    print(f"ROI: {summary['roi']:.3f}")
    print(f"Q-table final size: {len(agent.q_table)} states")
    print(f"\nResults saved to:")
    print(f"  Test log: {test_filename}")
    print(f"  Training log: {training_filename}")
    
    # Generate plots
    if not args.no_plots:
        print("Generating plots...")
        plot_rewards_over_time(test_log, plot_save_path)
        plot_action_distribution(test_log, plot_save_path)
        plot_training_progress(training_log, plot_save_path)
        print(f"Plots saved in: {plots_run_dir}")

if __name__ == "__main__":
    main()