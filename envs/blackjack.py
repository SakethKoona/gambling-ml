{"id":"59838","variant":"standard","title":"RL-Friendly Blackjack Environment with Policies"}
import random
from envs.utils import card_value, hand_value, Action
import pandas as pd

class BlackjackEnv:
    def __init__(self, n_decks=1):
        self.n_decks = n_decks
        self.deck = []
        self.player_hands = []
        self.dealer_cards = []
        self.done = False
        self.current_hand_idx = 0
        self.bets = []
        self.reward = 0

        # Stats tracking
        self.total_winnings = 0
        self.total_bets = 0
        self.rounds_played = 0
        self.rounds_won = 0
        self.rounds_lost = 0
        self.rounds_push = 0

    def _init_deck(self):
        self.deck = self.n_decks * ([c for c in range(1, 14)] * 4)
        random.shuffle(self.deck)

    def _deal_card(self):
        if not self.deck:
            self._init_deck()
        return self.deck.pop()

    def reset(self):
        self.done = False
        self.reward = 0
        self.current_hand_idx = 0

        self._init_deck()
        self.player_hands = [[self._deal_card(), self._deal_card()]]
        self.dealer_cards = [self._deal_card(), self._deal_card()]
        self.bets = [1]

        return self._get_state()

    def _get_state(self):
        hand = self.player_hands[self.current_hand_idx]
        player_total, is_soft = hand_value(hand)
        dealer_upcard_val = card_value(self.dealer_cards[0])
        can_double = int(len(hand) == 2)
        can_split = int(len(hand) == 2 and hand[0] == hand[1])
        return (player_total, int(is_soft), dealer_upcard_val, can_double, can_split)

    def get_valid_actions(self):
        hand = self.player_hands[self.current_hand_idx]
        valid = [Action.HIT, Action.STAND]
        if len(hand) == 2:
            valid.append(Action.DOUBLE_DOWN)
            if hand[0] == hand[1]:
                valid.append(Action.SPLIT)
        return valid

    def step(self, action: Action):
        if self.done:
            return None, self.reward, self.done, {}

        hand = self.player_hands[self.current_hand_idx]
        bet = self.bets[self.current_hand_idx]

        # Handle invalid action gracefully
        if action not in self.get_valid_actions():
            self.reward = -10 * bet  # large negative reward
            self.done = True
            return None, self.reward, self.done, {"invalid_action": True}

        match action:
            case Action.HIT:
                hand.append(self._deal_card())
                player_total, _ = hand_value(hand)
                if player_total > 21:
                    self.current_hand_idx += 1
                    if self.current_hand_idx >= len(self.player_hands):
                        self.done = True
                        self._play_dealer()
                        self._finalize_round()
            case Action.STAND:
                self.current_hand_idx += 1
                if self.current_hand_idx >= len(self.player_hands):
                    self.done = True
                    self._play_dealer()
                    self._finalize_round()
            case Action.DOUBLE_DOWN:
                self.bets[self.current_hand_idx] *= 2
                hand.append(self._deal_card())
                self.current_hand_idx += 1
                if self.current_hand_idx >= len(self.player_hands):
                    self.done = True
                    self._play_dealer()
                    self._finalize_round()
            case Action.SPLIT:
                card1, card2 = hand
                self.player_hands[self.current_hand_idx] = [card1, self._deal_card()]
                self.player_hands.insert(self.current_hand_idx + 1, [card2, self._deal_card()])
                self.bets.insert(self.current_hand_idx + 1, bet)
            case _:
                # Fallback in case of unexpected input
                self.reward = -10 * bet
                self.done = True
                return None, self.reward, self.done, {"invalid_action": True}

        next_state = None if self.done else self._get_state()
        return next_state, self.reward, self.done, {}

    def _play_dealer(self):
        total, _ = hand_value(self.dealer_cards)
        while total < 17:
            self.dealer_cards.append(self._deal_card())
            total, _ = hand_value(self.dealer_cards)

    def _compare_hand(self, hand, bet):
        player_total, _ = hand_value(hand)
        dealer_total, _ = hand_value(self.dealer_cards)

        if player_total > 21:
            return -bet
        if dealer_total > 21:
            return bet
        if player_total > dealer_total:
            return bet
        elif player_total < dealer_total:
            return -bet
        else:
            return 0

    def _calculate_total_reward(self):
        total_reward = 0
        for hand, bet in zip(self.player_hands, self.bets):
            total_reward += self._compare_hand(hand, bet)
        return total_reward

    def _finalize_round(self):
        self.reward = self._calculate_total_reward()
        self.total_winnings += self.reward
        self.total_bets += sum(self.bets)
        self.rounds_played += 1

        if self.reward > 0:
            self.rounds_won += 1
        elif self.reward < 0:
            self.rounds_lost += 1
        else:
            self.rounds_push += 1


# -----------------------------
# Simple baseline policies
# -----------------------------
def simple_policy(state):
    player_total, is_soft, dealer_upcard, can_double, can_split = state
    if can_split and player_total in [16, 20]:
        return Action.SPLIT
    if player_total <= 11 and can_double:
        return Action.DOUBLE_DOWN
    if player_total <= 11:
        return Action.HIT
    elif 12 <= player_total <= 16:
        if dealer_upcard >= 7:
            return Action.HIT
        else:
            return Action.STAND
    else:
        return Action.STAND


def stochastic_policy(state):
    player_total, is_soft, dealer_upcard, can_double, can_split = state

    p_hit, p_stand = 0.0, 0.0
    if player_total <= 11:
        p_hit, p_stand = 0.9, 0.1
    elif 12 <= player_total <= 16:
        if dealer_upcard >= 7:
            p_hit, p_stand = 0.7, 0.3
        else:
            p_hit, p_stand = 0.3, 0.7
    else:
        p_hit, p_stand = 0.1, 0.9

    if can_split and player_total in [16, 20]:
        return Action.SPLIT
    if can_double and player_total <= 11 and random.random() < 0.5:
        return Action.DOUBLE_DOWN

    return Action.HIT if random.random() < p_hit else Action.STAND


# -----------------------------
# Dataset generator
# -----------------------------
def generate_dataset(n_rounds=50000, seed=0):
    random.seed(seed)
    env = BlackjackEnv(n_decks=4)
    records = []

    for _ in range(n_rounds):
        state = env.reset()
        done = False
        while not done:
            action = simple_policy(state)
            next_state, reward, done, info = env.step(action)

            records.append({
                "player_total": state[0],
                "is_soft": state[1],
                "dealer_upcard": state[2],
                "can_double": state[3],
                "can_split": state[4],
                "action": int(action),
                "final_reward": reward if done else None,
                "total_winnings": env.total_winnings,
                "total_bets": env.total_bets,
                "rounds_played": env.rounds_played,
                "rounds_won": env.rounds_won,
                "rounds_lost": env.rounds_lost,
                "rounds_push": env.rounds_push
            })

            state = next_state

    df = pd.DataFrame(records)
    df = df.dropna(subset=["final_reward"])
    df["final_reward"] = df["final_reward"].astype(int)
    return df


# -----------------------------
# Run example
# -----------------------------
if __name__ == "__main__":
    df = generate_dataset(n_rounds=1000)
    print(df.head())
    print(df["action"].value_counts())
    print("Total winnings:", df["total_winnings"].iloc[-1])
    print("Rounds played:", df["rounds_played"].iloc[-1])
    print("Rounds won/lost/push:", df["rounds_won"].iloc[-1], df["rounds_lost"].iloc[-1], df["rounds_push"].iloc[-1])
