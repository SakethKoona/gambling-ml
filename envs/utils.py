from enum import IntEnum

class Action(IntEnum):
    STAND = 0
    HIT = 1
    DOUBLE_DOWN = 2
    SPLIT = 3


def card_value(card):
    """
    card: int from 1–13 where 1 = Ace, 11 = J, 12 = Q, 13 = K
    We map:
      2–10 -> face value
      J, Q, K -> 10
      A -> 11 (handled with soft logic)
    """
    if card == 1:
        return 11
    if card >= 10:
        return 10
    return card


def hand_value(cards):
    """
    Returns (total, is_soft)
    total: best total <= 21 if possible by adjusting Aces
    is_soft: True if hand contains at least one Ace counted as 11
    """
    total = 0
    aces = 0
    for c in cards:
        v = card_value(c)
        if v == 11:
            aces += 1
        total += v

    is_soft = False
    # Adjust for Aces if bust
    while total > 21 and aces > 0:
        total -= 10  # convert one Ace from 11 -> 1
        aces -= 1

    if aces > 0:
        is_soft = True

    return total, is_soft
