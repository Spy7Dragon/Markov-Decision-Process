import math
import numpy as np
import scipy.spatial.distance as distance
import QLearningAgent

X, Y, Z = range(0, 3)
NONE, LEFT, RIGHT = range(0, 3)
NORTH, EAST, SOUTH, WEST = range(0, 4)
dimensions = 99


class SkiJumper:
    states = range(10000)
    jump_feet = 1000
    actions = [NONE, LEFT, RIGHT]
    landed = False
    directions = [NORTH, EAST, SOUTH, WEST]
    rounds = 1
    saved_landscape = None
    mountain_height = 25000
    jumper_height = 100

    def __init__(self):
        self.facing = NORTH
        self.interim_direction = NORTH
        self.accel_direction = NORTH
        self.center = (50, 50)
        self.position = (50, 0, SkiJumper.mountain_height + SkiJumper.jumper_height)
        self.landscape = []
        self.state = self.create_state(self.position)
        self.start_state = self.state
        if SkiJumper.saved_landscape is None:
            for i in range(0, dimensions):
                self.landscape.append([])
                for j in range(0, dimensions):
                    dist = distance.euclidean([i, j],
                                              [self.center[X], self.center[Y]])
                    alt = SkiJumper.mountain_height - dist * (SkiJumper.mountain_height / 70)
                    self.landscape[i].append(alt)
            SkiJumper.saved_landscape = self.landscape
        else:
            self.landscape = SkiJumper.saved_landscape

    def get_state(self):
        return self.state

    def get_reward(self, position, action):
        reward = -1
        if action is not NONE:
            reward = reward - 1
        landed = False
        if 0 <= position[X] <= 98 and 0 <= position[Y] <= 98:
            landed = position[Z] < self.landscape[position[X]][position[Y]]
        else:
            reward -= 10000
        if landed:
            reward = position[Z]
        return reward

    def get_next_direction(self, action):
        direction = self.facing

        if action == LEFT:
            direction = direction - 1
        elif action == RIGHT:
            direction = direction + 1

        if direction == -1:
            direction = WEST
        elif direction == 4:
            direction = NORTH
        return direction

    @staticmethod
    def get_next_position(position, accel_direction):
        next_position = [0, 0, 0]
        if accel_direction == EAST:
            next_position[X] = position[X] + 1
            next_position[Y] = position[Y]
        elif accel_direction == WEST:
            next_position[X] = position[X] - 1
            next_position[Y] = position[Y]
        elif accel_direction == NORTH:
            next_position[X] = position[X]
            next_position[Y] = position[Y] + 1
        elif accel_direction == SOUTH:
            next_position[X] = position[X]
            next_position[Y] = position[Y] - 1
        next_position[Z] = position[Z] - 1

        if next_position[X] < 0:
            next_position[X] = 0
        elif next_position[X] >= dimensions:
            next_position[X] = dimensions - 1
        if next_position[Y] < 0:
            next_position[Y] = 0
        elif next_position[Y] >= dimensions:
            next_position[Y] = dimensions - 1
        if next_position[Z] < 0:
            next_position[Z] = 0
        return next_position

    def create_state(self, position):
        return position[X] * 100 + position[Y]

    def get_position(self, state):
        x = int(state / 100)
        y = int(state % 100)
        z = self.position[Z]
        position = [x, y, z]
        return position

    def get_next_state(self, state, action):
        position = self.get_position(state)
        self.accel_direction = self.interim_direction
        self.interim_direction = self.facing
        self.facing = self.get_next_direction(action)
        next_position = self.get_next_position(position, self.accel_direction)
        next_state = self.create_state(next_position)
        return next_state

    def get_next_positions(self):
        positions = []
        for dir in SkiJumper.directions:
            position = self.get_next_position(self.position, dir)
            positions.append(position)
        return positions

    def get_probabilities(self, action):
        return None

    def get_best_action(self):
        best_action = 0
        best_expected_value = float('-inf')
        for action in SkiJumper.actions:
            expected_value_sum = 0.0
            for next_pos in self.get_next_positions():
                expected_value = np.sum(0.25 * self.get_reward(next_pos, action))
                expected_value_sum += expected_value
            if expected_value_sum > best_expected_value:
                best_action = action
                best_expected_value = expected_value_sum
        return best_action

    def set_state(self, state):
        self.state = state
        position = self.get_position(state)
        position[Z] -= 1
        self.position = position
        self.landed = position[Z] < self.landscape[position[X]][position[Y]]

    @staticmethod
    def get_action_string(action):
        return str(SkiJumper.actions[action])

    @staticmethod
    def policy_iteration(model):
        P = np.full(len(model.states), 0)
        U = np.full(len(model.states), 0.0)
        finished = False
        gamma = 0.99
        iterations = 0
        while not finished:
            changed = False
            highest_z = SkiJumper.mountain_height + SkiJumper.jumper_height
            for i in range(0, 2 * SkiJumper.jumper_height):
                next_z = highest_z - i
                model.position = (model.position[X],
                                  model.position[Y],
                                  next_z)
                for state in model.states:
                    state_pos = model.get_position(state)
                    reward = model.get_reward(state_pos, NONE)
                    expected_value_sum = 0.0
                    for next_pos in model.get_next_positions():
                        next_state = model.create_state(next_pos)
                        expected_value = np.sum(0.25 * U[next_state])
                        expected_value_sum += expected_value
                    U[state] = reward + gamma * expected_value_sum
                for state in model.states:
                    max_action = 0
                    max_ev_sum = float('-inf')
                    for action in model.actions:
                        expected_value_sum = 0.0
                        for next_pos in model.get_next_positions():
                            next_state = model.create_state(next_pos)
                            expected_value = np.sum(0.25 * U[next_state])
                            expected_value_sum += expected_value
                        if expected_value_sum > max_ev_sum:
                            max_action = action
                            max_ev_sum = expected_value_sum
                    if max_ev_sum > U[state]:
                        if P[state] != max_action:
                            P[state] = max_action
                            changed = True
            iterations += 1
            if not changed:
                finished = True
        print "Iterations: " + str(iterations)
        # for i in range(0, len(P)):
        #     action = P[i]
        #     print "State: " + str(i) + " ,Action: " + model.get_action_string(action)
        # print U
        return P

    @staticmethod
    def value_iteration(model, max_iter=10):
        U = np.full(len(model.states), 0.0)
        finished = False
        threshold = 10
        gamma = 0.99
        iterations = 0
        while not finished and iterations < max_iter:
            difference = 0.0
            highest_z = SkiJumper.mountain_height + SkiJumper.jumper_height
            for i in range(0, 2 * SkiJumper.jumper_height):
                next_z = highest_z - i
                model.position = (model.position[X],
                                  model.position[Y],
                                  next_z)
                for state in model.states:
                    max_qsa = float('-inf')
                    old_vsa = U[state]
                    state_pos = model.get_position(state)
                    reward = model.get_reward(state_pos, NONE)
                    for action in model.actions:
                        expected_value_sum = 0.0
                        for next_pos in model.get_next_positions():
                            next_state = model.create_state(next_pos)
                            expected_value = np.sum(0.25 * U[next_state])
                            expected_value_sum += expected_value
                        qsa = gamma * expected_value_sum + reward
                        if qsa > max_qsa:
                            max_qsa = qsa
                    U[state] = max_qsa
                    ind_diff = math.pow(old_vsa - max_qsa, 2)
                    difference += ind_diff
            iterations += 1
            if difference < threshold:
                finished = True
        print "Iterations: " + str(iterations)
        print U

    @staticmethod
    def qlearner_iteration(model, epochs=10000):
        discount = 0.9
        alpha = 0.5
        random_rate = 0.1
        learner = QLearningAgent.QLearningAgent(len(model.states), len(model.actions),
                                                discount_rate=discount, learning_rate=alpha, random_rate=random_rate)

        for i in range(0, epochs):
            model = SkiJumper()
            action = model.get_best_action()
            while not model.landed:
                next_state = model.get_next_state(learner.state, action)
                next_state_pos = model.get_position(next_state)
                reward = model.get_reward(next_state_pos, action)
                action = learner.get_best_action(next_state, reward)
                model.set_state(next_state)
                learner.set_state(next_state)
        return learner