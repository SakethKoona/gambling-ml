import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
from collections import deque, defaultdict
from datetime import datetime
from envs.utils import Action
import matplotlib.pyplot as plt
import pandas as pd

# Assuming the BlackjackEnv is imported from your file
# from your_file import BlackjackEnv
from envs.blackjack import BlackjackEnv

class DQN(nn.Module):
    """Deep Q-Network for Blackjack"""
    def __init__(self, state_dim=5, action_dim=4, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for Blackjack"""
    def __init__(self, state_dim=5, action_dim=4, lr=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
    def select_action(self, state, valid_actions, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Mask invalid actions
            masked_q = np.full(self.action_dim, -np.inf)
            for action in valid_actions:
                masked_q[int(action)] = q_values[int(action)]
            
            return Action(np.argmax(masked_q))
    
    def train_step(self, batch_size=128):
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors - ensure proper numpy array conversion
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Target Q values - FIXED: only compute for non-terminal states
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # For terminal states, target is just the reward (no future value)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_dqn(env, agent, n_episodes=10000, batch_size=128, 
              target_update_freq=100, eval_freq=500, eval_episodes=100, save_dir="plots"):
    """Train DQN agent"""
    training_stats = {
        'episode': [],
        'total_reward': [],
        'epsilon': [],
        'loss': [],
        'avg_reward': [],
        'eval_roi': [],
        'eval_win_rate': []
    }
    
    print(f"Training on device: {agent.device}")
    print("=" * 60)
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        losses = []
        steps_in_episode = 0
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=True)
            next_state, reward, done, _ = env.step(action)
            
            # CRITICAL FIX: Store the actual next_state, not the current state when done
            # Use a zero/dummy state for terminal states instead
            if done and next_state is None:
                # Terminal state - use current state as placeholder (won't be used due to done=True)
                store_next_state = state
            else:
                store_next_state = next_state if next_state is not None else state
            
            # Store transition
            agent.replay_buffer.push(
                state, int(action), reward, 
                store_next_state, 
                done
            )
            
            # Train after every step (not just at end of episode)
            loss = agent.train_step(batch_size)
            if loss is not None:
                losses.append(loss)
            
            if done:
                episode_reward = reward  # Final reward of the round
            
            state = next_state
            steps_in_episode += 1
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Log training stats
        training_stats['episode'].append(episode)
        training_stats['total_reward'].append(episode_reward)
        training_stats['epsilon'].append(agent.epsilon)
        training_stats['loss'].append(np.mean(losses) if losses else 0)
        
        # Evaluation
        if episode % eval_freq == 0:
            eval_results = evaluate_agent(env, agent, n_episodes=eval_episodes)
            training_stats['avg_reward'].append(eval_results['avg_reward'])
            training_stats['eval_roi'].append(eval_results['roi'])
            training_stats['eval_win_rate'].append(eval_results['win_rate'])
            
            print(f"Episode {episode}/{n_episodes}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Loss: {np.mean(losses) if losses else 0:.4f}")
            print(f"  Replay Buffer: {len(agent.replay_buffer)}")
            print(f"  Eval ROI: {eval_results['roi']:.2f}%")
            print(f"  Eval Win Rate: {eval_results['win_rate']:.2f}%")
            print(f"  Eval Avg Reward: {eval_results['avg_reward']:.3f}")
            print("-" * 60)
    
    return training_stats


def evaluate_agent(env, agent, n_episodes=1000, save_detailed_log=False):
    """Evaluate trained agent"""
    total_rewards = []
    detailed_episodes = [] if save_detailed_log else None
    
    # Reset env stats at the start
    env.total_winnings = 0
    env.total_bets = 0
    env.rounds_played = 0
    env.rounds_won = 0
    env.rounds_lost = 0
    env.rounds_push = 0
    
    for ep_idx in range(n_episodes):
        state = env.reset()
        done = False
        
        if save_detailed_log:
            episode_data = {
                "episode": ep_idx,
                "steps": [],
                "final_reward": 0,
                "player_hands": [],
                "dealer_cards": [],
                "total_bets": 0
            }
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            
            if save_detailed_log:
                step_info = {
                    "state": {
                        "player_total": state[0],
                        "is_soft": bool(state[1]),
                        "dealer_upcard": state[2],
                        "can_double": bool(state[3]),
                        "can_split": bool(state[4])
                    },
                    "action": int(action),
                    "action_name": action.name
                }
                episode_data["steps"].append(step_info)
            
            next_state, reward, done, _ = env.step(action)
            
            if save_detailed_log and done:
                step_info["reward"] = reward
                episode_data["final_reward"] = reward
                episode_data["player_hands"] = [hand.copy() for hand in env.player_hands]
                episode_data["dealer_cards"] = env.dealer_cards.copy()
                episode_data["total_bets"] = sum(env.bets)
            
            state = next_state
        
        if save_detailed_log:
            detailed_episodes.append(episode_data)
        
        total_rewards.append(reward)
    
    results = {
        'total_reward': sum(total_rewards),
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'total_winnings': env.total_winnings,
        'total_bets': env.total_bets,
        'roi': (env.total_winnings / env.total_bets * 100) if env.total_bets > 0 else 0,
        'win_rate': (env.rounds_won / env.rounds_played * 100) if env.rounds_played > 0 else 0,
        'loss_rate': (env.rounds_lost / env.rounds_played * 100) if env.rounds_played > 0 else 0,
        'push_rate': (env.rounds_push / env.rounds_played * 100) if env.rounds_played > 0 else 0,
        'rounds_won': env.rounds_won,
        'rounds_lost': env.rounds_lost,
        'rounds_push': env.rounds_push
    }
    
    if save_detailed_log:
        results['episodes'] = detailed_episodes
    
    return results


def plot_training_stats(training_stats, save_dir):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Epsilon decay
    axes[0, 0].plot(training_stats['episode'], training_stats['epsilon'])
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Epsilon')
    axes[0, 0].set_title('Exploration Rate Over Time')
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(training_stats['episode'], training_stats['loss'])
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True)
    
    # ROI
    eval_episodes = [training_stats['episode'][i] for i in range(0, len(training_stats['episode']), 500)]
    axes[1, 0].plot(eval_episodes, training_stats['eval_roi'])
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('ROI (%)')
    axes[1, 0].set_title('Evaluation ROI')
    axes[1, 0].grid(True)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Win Rate
    axes[1, 1].plot(eval_episodes, training_stats['eval_win_rate'])
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Win Rate (%)')
    axes[1, 1].set_title('Evaluation Win Rate')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to '{save_dir}/training_stats.png'")


def plot_action_distribution(episodes, save_dir):
    """Plot action distribution from test episodes"""
    action_counts = {action.name: 0 for action in Action}
    
    for ep in episodes:
        for step in ep["steps"]:
            action_counts[step["action_name"]] += 1
    
    plt.figure(figsize=(10, 6))
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    
    plt.bar(actions, counts, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    plt.title('Action Distribution (DQN)')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/action_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cumulative_reward(episodes, save_dir, roi):
    """Plot cumulative reward over test episodes"""
    rewards = [ep["final_reward"] for ep in episodes]
    cumulative = np.cumsum(rewards)
    
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative, color='darkblue', linewidth=2)
    plt.title(f"DQN Cumulative Reward (ROI: {roi:.3f}%)")
    plt.xlabel("Episode")
    plt.ylabel("Total Winnings")
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cumulative_reward.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_results(results, title="Evaluation Results"):
    """Print evaluation results in a formatted way"""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    print(f"Total Reward:        {results['total_reward']:>10.2f}")
    print(f"Average Reward:      {results['avg_reward']:>10.3f} ± {results['std_reward']:.3f}")
    print(f"Total Winnings:      {results['total_winnings']:>10.2f}")
    print(f"Total Bets:          {results['total_bets']:>10.2f}")
    print(f"ROI:                 {results['roi']:>10.2f}%")
    print("-" * 60)
    print(f"Win Rate:            {results['win_rate']:>10.2f}%")
    print(f"Loss Rate:           {results['loss_rate']:>10.2f}%")
    print(f"Push Rate:           {results['push_rate']:>10.2f}%")
    print("-" * 60)
    print(f"Rounds Won:          {results['rounds_won']:>10}")
    print(f"Rounds Lost:         {results['rounds_lost']:>10}")
    print(f"Rounds Push:         {results['rounds_push']:>10}")
    print("=" * 60)


def save_test_log(results, training_params, timestamp, save_dir):
    """Save detailed test log as JSON and CSV"""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Prepare test log
    test_log = {
        "episodes": results.get('episodes', []),
        "summary": {
            "total_reward": results['total_reward'],
            "average_reward": results['avg_reward'],
            "total_winnings": results['total_winnings'],
            "total_bets": results['total_bets'],
            "roi": results['roi'],
            "win_rate": results['win_rate'] / 100,
            "rounds_won": results['rounds_won'],
            "rounds_lost": results['rounds_lost'],
            "rounds_push": results['rounds_push']
        },
        "metadata": {
            "timestamp": timestamp,
            "algorithm": "dqn",
            **training_params
        }
    }
    
    # Save JSON
    json_filename = f"logs/test_results_dqn_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(test_log, f, indent=2)
    print(f"JSON log saved to '{json_filename}'")
    
    # Save CSV if episodes are available
    if 'episodes' in results and results['episodes']:
        csv_rows = []
        for episode_idx, ep_data in enumerate(results['episodes']):
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
        csv_filename = f"logs/dqn_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"CSV data saved to '{csv_filename}'")
    """Print evaluation results in a formatted way"""
    print("\n" + "=" * 60)
    print("=" * 60)
    print(f"Total Reward:        {results['total_reward']:>10.2f}")
    print(f"Average Reward:      {results['avg_reward']:>10.3f} ± {results['std_reward']:.3f}")
    print(f"Total Winnings:      {results['total_winnings']:>10.2f}")
    print(f"Total Bets:          {results['total_bets']:>10.2f}")
    print(f"ROI:                 {results['roi']:>10.2f}%")
    print("-" * 60)
    print(f"Win Rate:            {results['win_rate']:>10.2f}%")
    print(f"Loss Rate:           {results['loss_rate']:>10.2f}%")
    print(f"Push Rate:           {results['push_rate']:>10.2f}%")
    print("-" * 60)
    print(f"Rounds Won:          {results['rounds_won']:>10}")
    print(f"Rounds Lost:         {results['rounds_lost']:>10}")
    print(f"Rounds Push:         {results['rounds_push']:>10}")
    print("=" * 60)


if __name__ == "__main__":
    # Import your BlackjackEnv here
    # from your_blackjack_file import BlackjackEnv
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create timestamp and directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"plots/dqn_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Training parameters
    training_params = {
        "n_episodes": 10000,
        "batch_size": 128,
        "target_update_freq": 100,
        "eval_freq": 500,
        "eval_episodes": 100,
        "test_episodes": 10000,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "hidden_dim": 128
    }
    
    # Initialize environment and agent
    env = BlackjackEnv(n_decks=4)
    agent = DQNAgent(
        state_dim=5,
        action_dim=4,
        lr=training_params["learning_rate"],
        gamma=training_params["gamma"],
        epsilon_start=training_params["epsilon_start"],
        epsilon_end=training_params["epsilon_end"],
        epsilon_decay=training_params["epsilon_decay"]
    )
    
    # Train the agent
    print("=" * 60)
    print("DQN TRAINING")
    print("=" * 60)
    print(f"Training episodes:     {training_params['n_episodes']:,}")
    print(f"Batch size:            {training_params['batch_size']}")
    print(f"Learning rate:         {training_params['learning_rate']}")
    print(f"Discount factor:       {training_params['gamma']}")
    print(f"Epsilon decay:         {training_params['epsilon_decay']}")
    print("=" * 60)
    
    training_stats = train_dqn(
        env, agent,
        n_episodes=training_params["n_episodes"],
        batch_size=training_params["batch_size"],
        target_update_freq=training_params["target_update_freq"],
        eval_freq=training_params["eval_freq"],
        eval_episodes=training_params["eval_episodes"],
        save_dir=save_dir
    )
    
    # Plot training statistics
    plot_training_stats(training_stats, save_dir)
    
    # Final evaluation with detailed logging
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    print(f"Running {training_params['test_episodes']:,} test episodes...")
    
    final_results = evaluate_agent(env, agent, n_episodes=training_params["test_episodes"], save_detailed_log=True)
    print_results(final_results, f"Final Test Results ({training_params['test_episodes']:,} episodes)")
    
    # Save detailed logs
    save_test_log(final_results, training_params, timestamp, save_dir)
    
    # Generate additional plots
    if 'episodes' in final_results:
        plot_action_distribution(final_results['episodes'], save_dir)
        plot_cumulative_reward(final_results['episodes'], save_dir, final_results['roi'])
    
    # Save the trained model
    model_filename = f"logs/dqn_blackjack_model_{timestamp}.pth"
    torch.save({
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'training_params': training_params,
        'final_epsilon': agent.epsilon
    }, model_filename)
    print(f"Model saved to '{model_filename}'")
    
    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)
    print(f"Plots directory:       {save_dir}/")
    print(f"Training stats:        {save_dir}/training_stats.png")
    print(f"Action distribution:   {save_dir}/action_distribution.png")
    print(f"Cumulative reward:     {save_dir}/cumulative_reward.png")
    print(f"Model checkpoint:      {model_filename}")
    print("=" * 60)