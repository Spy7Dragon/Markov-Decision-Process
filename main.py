import math
import numpy as np

import QLearningAgent
import StrengthTester
import SkiJumper


def ski_jumper_iteration(learner, verbose=False):
    model = SkiJumper.SkiJumper()
    learner.set_state(model.state)
    model.get_best_action()
    action = SkiJumper.NONE
    while not model.landed:
        if verbose: print "Action: " + model.get_action_string(action)
        next_state = model.get_next_state(learner.s, action)
        if verbose: print "Next State: " + str(next_state)
        next_state_pos = model.get_position(next_state)
        reward = model.get_reward(next_state_pos, action)
        action = learner.get_best_action(next_state, reward)
        model.set_state(next_state)
        learner.set_state(next_state)


def run_strength_tester_model(model):
    learner = model.qlearner_iteration(model, 100000)
    start_index = 0
    start_state = start_index + 1
    state = start_state
    print "Start State: " + str(start_state)
    total_reward = 0
    action = learner.set_state(state)
    for round in range(0, model.rounds):
        print "Round: " + str(round + 1)
        print "Action: " + model.get_action_string(action)
        next_state = model.get_next_state(learner.s, action)
        next_state_pos = model.get_state_position(next_state)
        print "Next State: " + str(next_state_pos[0] + 1)
        reward = model.get_reward(next_state_pos[0] + 1)
        action = learner.get_best_action(next_state, reward)
        total_reward += reward
    print "Total Reward: " + str(total_reward)

    policy = model.policy_iteration(model)
    start_index = 0
    start_state = start_index + 1
    state = start_index
    print "Start State: " + str(start_state)
    total_reward = 0
    for round in range(0, model.rounds):
        print "Round: " + str(round + 1)
        state_pos = model.get_state_position(state)
        action = policy[state_pos]
        print "Action: " + model.get_action_string(action)
        next_state = model.get_next_state(state, action)
        next_state_pos = model.get_state_position(next_state)
        print "Next State: " + str(next_state_pos[0] + 1)
        reward = model.get_reward(next_state_pos[0] + 1)
        state = model.get_state(round + 1, next_state_pos)
        total_reward += reward
    print "Total Reward: " + str(total_reward)

    model.value_iteration(model)
    start_index = 0
    start_state = start_index + 1
    state = start_state
    print "Start State: " + str(start_state)
    total_reward = 0
    for round in range(0, model.rounds):
        print "Round: " + str(round + 1)
        action = model.get_best_action(state)
        print "Action: " + model.get_action_string(action)
        next_state = model.get_next_state(state, action)
        next_state_pos = model.get_state_position(next_state)
        print "Next State: " + str(next_state_pos[0] + 1)
        reward = model.get_reward(next_state_pos[0] + 1)
        state = model.get_state(round + 1, next_state_pos)
        total_reward += reward
    print "Total Reward: " + str(total_reward)


def run_ski_jumper_model(model):
    learner = model.qlearner_iteration(model, 100)
    print "Start State: " + str(model.state)
    total_reward = 0
    action = learner.set_state(model.state)
    while not model.landed:
        print "Altitude: " + str(model.position[SkiJumper.Z])
        print "Action: " + model.get_action_string(action)
        next_state = model.get_next_state(model.state, action)
        next_position = model.get_position(next_state)
        print "Next State: " + str(next_state)
        reward = model.get_reward(next_position, action)
        action = learner.get_best_action(next_state, reward)
        model.set_state(next_state)
        total_reward += reward
    print "Total Reward: " + str(total_reward)

    policy = model.policy_iteration(model)
    model = SkiJumper.SkiJumper()
    # print "Start State: " + str(model.state)
    total_reward = 0
    while not model.landed:
        # print "Altitude: " + str(model.position[SkiJumper.Z])
        action = policy[model.state]
        # print "Action: " + model.get_action_string(action)
        next_state = model.get_next_state(model.state, action)
        # print "Next State: " + str(next_state)
        next_position = model.get_position(next_state)
        reward = model.get_reward(next_position, action)
        model.set_state(next_state)
        total_reward += reward
    print "Total Reward: " + str(total_reward)

    model.value_iteration(model)
    model = SkiJumper.SkiJumper()
    # print "Start State: " + str(model.state)
    total_reward = 0
    while not model.landed:
        # print "Altitude: " + str(model.position[SkiJumper.Z])
        action = model.get_best_action()
        # print "Action: " + model.get_action_string(action)
        next_state = model.get_next_state(model.state, action)
        next_position = model.get_position(next_state)
        # print "Next State: " + str(next_state)
        reward = model.get_reward(next_position, action)
        model.set_state(next_state)
        total_reward += reward
    print "Total Reward: " + str(total_reward[0])


if __name__ == "__main__":
    strength_tester_model = StrengthTester.StrengthTester()
    run_strength_tester_model(strength_tester_model)
    ski_jumper_mdp = SkiJumper.SkiJumper()
    run_ski_jumper_model(ski_jumper_mdp)


