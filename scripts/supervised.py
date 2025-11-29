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

    # Save to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    filename = f"{logs_dir}/test_results_{args.model_type}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(test_log, f, indent=2)

    print(f"Average reward over {n_test_episodes} test episodes: {total_reward / n_test_episodes}: {total_reward}")
    print(f"Test results saved to: {filename}")
    print(f"Win rate: {test_log['summary']['win_rate']:.3f}")
    print(f"ROI: {test_log['summary']['roi']:.3f}")



