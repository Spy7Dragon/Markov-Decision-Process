import numpy as np
import random


class QLearningAgent(object):
    def __init__(self, num_states=10, num_actions=2, learning_rate=0.2, discount_rate=0.9, \
                 random_rate=0.5, random_decay=0.99):
        self.num_actions = num_actions
        self.num_states = num_states
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.random_rate = random_rate
        self.random_decay = random_decay
        self.state = 0
        self.action = 0

        self.Q = np.zeros((num_states, num_actions))
        self.Q.fill(0.0)

    def set_state(self, state):
        self.state = state
        possible_actions = self.Q[self.state, :]
        action = np.argmax(possible_actions)
        if np.random.random() < self.random_rate:
            action = random.randint(0, self.num_actions - 1)

        self.action = action

        return action

    def get_best_action(self, next_state, reward):
        possible_next_actions = self.Q[next_state, :]
        best_next_action = np.argmax(possible_next_actions)
        self.Q[self.state, self.action] = \
            (1.0 - self.learning_rate) * self.Q[self.state, self.action] \
            + self.learning_rate * (reward + self.discount_rate * self.Q[next_state, best_next_action])

        # Set new state
        self.state = next_state
        # Find action
        possible_next_actions = self.Q[next_state, :]
        next_action = np.argmax(possible_next_actions)
        if np.random.random() < self.random_rate:
            next_action = random.randint(0, self.num_actions - 1)
        self.action = next_action

        self.random_rate *= self.random_decay

        return next_action

