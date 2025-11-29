# qlearning.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import random
from envs.utils import Action
from envs.blackjack import BlackjackEnv 

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # State space: player_total 4–21, is_soft 0/1, dealer_upcard 1–10, can_double 0/1, can_split 0/1
        self.player_totals = list(range(4, 22))
        self.soft_flags = [0, 1]
        self.dealer_upcards = list(range(1, 11))
        self.double_flags = [0, 1]
        self.split_flags = [0, 1]

        # Action space: HIT=0, STAND=1, DOUBLE_DOWN=2, SPLIT=3
        self.n_actions = 4

        # Initialize Q-table as dict of dicts
        self.q_table = {}

    def get_state_key(self, state):
        return tuple(state)  # (player_total, is_soft, dealer_upcard, can_double, can_split)

    def choose_action(self, state):
        # Use epsilon greedy for action selection
        if random.random() < self.epsilon:
            return random.choice(range(self.n_actions))
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        return int(np.argmax(self.q_table[key]))

    def learn(self, state, action, reward, next_state, done):
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)

        if next_state is not None:
            next_key = self.get_state_key(next_state)
            if next_key not in self.q_table:
                self.q_table[next_key] = np.zeros(self.n_actions)
            target = reward + self.gamma * np.max(self.q_table[next_key])
        else:
            target = reward

        self.q_table[key][action] += self.lr * (target - self.q_table[key][action])

    def train(self, episodes=50000):
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(Action(action))
                self.learn(state, action, reward, next_state, done)
                state = next_state

            # Optional: decay epsilon
            if ep % 1000 == 0 and self.epsilon > 0.01:
                self.epsilon *= 0.995

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    env = BlackjackEnv(n_decks=4)
    agent = QLearningAgent(env)

    print("Training Q-learning agent...")
    agent.train(episodes=10000)

    # Test (one episode, which represents a single round of blackjack)
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(Action(action))
        total_reward += reward if done else 0
        state = next_state

    print("Test round reward:", total_reward)
