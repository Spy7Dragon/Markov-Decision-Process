import math
import numpy as np
import QLearningAgent


class StrengthTester:
    #mallot_types = SOFT, MEDIUM, HARD
    #HIT_STRENGTH = range(1, 11)

    states = range(110)
    rounds = 10
    actions = range(30)
    start_state = 1
    state_positions = range(10)

    def __init__(self):
        pass

    @staticmethod
    def get_state(round, position):
        state = round * 10 + position
        return state

    @staticmethod
    def get_state_round(state):
        round_index = int(state / 10)
        state_round = round_index
        return state_round

    @staticmethod
    def get_state_position(state):
        position_index = state % 10
        return position_index

    @staticmethod
    def get_reward(state):
        return state

    @staticmethod
    def get_next_state(state, action):
        state_round = StrengthTester.get_state_round(state)
        state_pos = StrengthTester.get_state_position(state)
        distribution = StrengthTester.get_probabilities(state_pos, action)
        next_index = np.random.choice(range(10), 1, p=distribution)
        next_pos = next_index + 1
        next_round = state_round + 1
        next_state = (next_round - 1) * 10 + (next_pos - 1)
        return next_state

    @staticmethod
    def get_probabilities(state_pos, action):
        mallet = int(action / 10)
        hit_strength = (action % 10 + 1) + mallet
        greater_radius = int(math.sqrt(hit_strength) + math.pow(mallet, 2))
        lesser_radius = np.min([hit_strength - 1, greater_radius])
        prob_count = np.repeat(0.0, 10)
        center = (state_pos + hit_strength) % 10
        greater_position = center
        lesser_position = center
        prob_count[center] = 1.0
        for i in range(0, greater_radius):
            greater_position = (greater_position + 1) % 10
            prob_count[greater_position] += 1.0
        for i in range(0, lesser_radius):
            lesser_position = (lesser_position - 1) % 10
            prob_count[lesser_position] += 1.0
        prob_count = prob_count / np.sum(prob_count)
        return prob_count

    @staticmethod
    def get_best_action(state):
        state_round = StrengthTester.get_state_round(state)
        state_pos = StrengthTester.get_state_position(state)
        best_action = 0
        best_expected_value = float('-inf')
        for action in StrengthTester.actions:
            probabilities = StrengthTester.get_probabilities(state_pos - 1, action)
            expected_value_sum = 0.0
            for i in range(0, len(probabilities)):
                expected_value = np.sum(probabilities[i] * StrengthTester.get_reward(i + 1))
                expected_value_sum += expected_value
            if expected_value_sum > best_expected_value:
                best_action = action
                best_expected_value = expected_value_sum
        return best_action

    @staticmethod
    def get_action_string(action):
        if isinstance(action, np.ndarray):
            action = action[0]
        mallets = ['SOFT', 'MEDIUM', 'HARD']
        strengths = range(1, 11)
        mallet = int(action / 10)
        strength = strengths[action % 10]
        string = mallets[mallet] + '-' + str(strength)
        return string

    @staticmethod
    def value_iteration(model):
        U = np.full(len(model.state_positions), 0.0)
        finished = False
        threshold = 0.01
        gamma = 0.99
        iterations = 0
        while not finished:
            difference = 0.0
            for round in range(0, model.rounds):
                if round == 0:
                    state = model.get_state(round, 0)
                    state_pos = model.get_state_position(state)
                    max_qsa = float('-inf')
                    old_vsa = U[state_pos]
                    reward = model.get_reward(state_pos + 1)
                    for action in model.actions:
                        probabilities = model.get_probabilities(state_pos, action)
                        expected_value_sum = 0.0
                        for i in range(0, len(probabilities)):
                            next_pos = i
                            expected_value = np.sum(probabilities[next_pos] * U[next_pos])
                            expected_value_sum += expected_value
                        qsa = gamma * expected_value_sum + reward
                        if qsa > max_qsa:
                            max_qsa = qsa
                    U[state_pos] = max_qsa
                    ind_diff = math.pow(old_vsa - max_qsa, 2)
                    difference += ind_diff
                else:
                    for state_pos in model.state_positions:
                        state = model.get_state(round, state_pos)
                        max_qsa = float('-inf')
                        old_vsa = U[state_pos]
                        reward = model.get_reward(state_pos + 1)
                        for action in model.actions:
                            probabilities = model.get_probabilities(state_pos, action)
                            expected_value_sum = 0.0
                            for i in range(0, len(probabilities)):
                                next_pos = i
                                expected_value = np.sum(probabilities[next_pos] * U[next_pos])
                                expected_value_sum += expected_value
                            qsa = gamma * expected_value_sum + reward
                            if qsa > max_qsa:
                                max_qsa = qsa
                        U[state_pos] = max_qsa
                        ind_diff = math.pow(old_vsa - max_qsa, 2)
                        difference += ind_diff
            iterations += 1
            if difference < threshold:
                finished = True
        print "Iterations: " + str(iterations)
        print U

    @staticmethod
    def policy_iteration(model):
        P = np.full(len(model.state_positions), 0)
        U = np.full(len(model.state_positions), 0.0)
        finished = False
        gamma = 0.5
        iterations = 0
        while not finished:
            changed = False
            for round in range(0, model.rounds):
                if round == 0:
                    state = model.get_state(round, 0)
                    state_pos = model.get_state_position(state)
                    reward = model.get_reward(state_pos + 1)
                    probabilities = model.get_probabilities(state_pos, P[state_pos])
                    expected_value_sum = 0.0
                    for i in range(0, len(probabilities)):
                        next_pos = i
                        expected_value = np.sum(probabilities[next_pos] * U[next_pos])
                        expected_value_sum += expected_value
                    U[state_pos] = reward + gamma * expected_value_sum
                    max_action = 0
                    max_ev_sum = float('-inf')
                    for action in model.actions:
                        probabilities = model.get_probabilities(state_pos, action)
                        expected_value_sum = 0.0
                        for i in range(0, len(probabilities)):
                            next_pos = i
                            expected_value = np.sum(probabilities[next_pos] * U[next_pos])
                            expected_value_sum += expected_value
                        if expected_value_sum > max_ev_sum:
                            max_action = action
                            max_ev_sum = expected_value_sum
                    if max_ev_sum > U[state_pos]:
                        if P[state_pos] != max_action:
                            P[state_pos] = max_action
                            changed = True
                else:
                    for state_pos in model.state_positions:
                        reward = model.get_reward(state_pos + 1)
                        probabilities = model.get_probabilities(state_pos, P[state_pos])
                        expected_value_sum = 0.0
                        for i in range(0, len(probabilities)):
                            next_pos = i
                            expected_value = np.sum(probabilities[next_pos] * U[next_pos])
                            expected_value_sum += expected_value
                        U[state_pos] = reward + gamma * expected_value_sum
                    for state_pos in model.state_positions:
                        max_action = 0
                        max_ev_sum = float('-inf')
                        for action in model.actions:
                            probabilities = model.get_probabilities(state_pos, action)
                            expected_value_sum = 0.0
                            for i in range(0, len(probabilities)):
                                next_pos = i
                                expected_value = np.sum(probabilities[next_pos] * U[next_pos])
                                expected_value_sum += expected_value
                            if expected_value_sum > max_ev_sum:
                                max_action = action
                                max_ev_sum = expected_value_sum
                        if max_ev_sum > U[state_pos]:
                            if P[state_pos] != max_action:
                                P[state_pos] = max_action
                                changed = True
            iterations += 1
            if not changed:
                finished = True
        print "Iterations: " + str(iterations)
        for i in range(0, len(P)):
            action = P[i]
            print "State: " + str(i + 1) + ", Action: " + model.get_action_string(action)
        print U
        return P

    @staticmethod
    def qlearner_iteration(model, epochs=10000):
        discount = 0.1
        alpha = 0.9
        random_rate = 0.9
        learner = QLearningAgent.QLearningAgent(len(model.states), len(model.actions),
                                                discount_rate=discount, learning_rate=alpha, random_rate=random_rate)
        for i in range(0, epochs):
            for round in range(0, model.rounds):
                if round == 0:
                    action = learner.set_state(0)
                next_state = model.get_next_state(learner.state, action)
                next_state_pos = model.get_state_position(next_state)
                reward = model.get_reward(next_state_pos)
                action = learner.get_best_action(next_state, reward)
                state = model.get_state(round + 1, next_state_pos)
                learner.set_state(state)
        return learner
