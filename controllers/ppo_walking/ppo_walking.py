"""ppo_walking controller."""

import numpy as np
import socket
import pickle
import time
import os

from controller import Robot, Field, Node, Supervisor, Motor
from environment import Environment

# create the robot instance
supervisor = Supervisor()
robot = supervisor

env = Environment(robot)

# create socket for communication with RL-agent
host = socket.gethostname()
port = 50005    # random unused port
client_socket = socket.socket()
client_socket.connect((host, port))

def get_action(state):
    """ Get prediction for new action from RL-agent. 
        Use state as input for actor network.
        State is send to the external RL-agent over socket. 
        Data is serialized and de-serialized using pickle.

    Args:
        state: the current state of the environment
    Returns:
        response[0]: the predicted action vector
        response[1]: the log probability of the predicted actions
        response[2]: the value function estimate
    """

    state = np.stack([state])
    data = {'method': 'action', 'state': state}
    data = pickle.dumps(data)

    client_socket.send(data)
    response = client_socket.recv(1024)
    response = pickle.loads(response)

    # action, log_probs, value
    return response[0], response[1], response[2]

def send_log_data(step, total_reward):
    """ Send all data to the RL-agent after episode is finished. 
    `   Data is then added to  tensorboard and agent is trained if enough 
    `   trajectories are gathered. 

    Args:
        step: number of steps executed in the episode
        total_reward: accumulated reward over episode
    """

    # prepare data as dictionary 
    batch_content = {
        'states': batch_state,
        'actions': batch_action,
        'returns': batch_discounted_returns,
        'old_log_probs':batch_old_log_probs,
        'values': batch_value,
        'next_values': batch_value[1:] + [0],
        'rewards': batch_rewards,
        'reward_info': reward_info
        }

    # serialize and store on disk (too large for socket)
    pickle.dump(batch_content, open('./vars/batch.p', "wb"))
    data = {'method': 'log', 'data': [step, total_reward]}
    data = pickle.dumps(data)
    client_socket.send(data)

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# counter for the current step in the episode, resets after max_steps
episode_step = 0
done = False
total_reward = 0

# initialize state
robot.step(timestep)
state = env.reset()

# complete trajectory
batch_state = []
batch_action = []
batch_old_log_probs = []
batch_value = []
batch_discounted_returns = []

# current episode
batch_rewards = []
reward_info = []

step = 0

skip = 4  # only get new action every {skip} steps, execute old action in other steps
reward_skip_step = 0

# Main loop
while robot.step(timestep) != -1: 

    # Lower action sampling frequency is implemented using action skip. 
    # Higher frequency in webots leads to higher inacurracies in physics simulation.
    if step % skip == 0:
        # Get action, probs, and value estimate from agent
        action, old_log_probs, value = get_action(state)

        # Perform step in environment 
        next_state, reward, done = env.step(action)

        # Remember reward
        reward += reward_skip_step
        reward_skip_step = 0
        total_reward += reward

        # Remember data for trajectory
        batch_state.append(state)
        batch_action.append(action)
        batch_old_log_probs.append(old_log_probs)
        batch_rewards.append(reward)
        batch_value.append(np.asscalar(value))

    else:
        # repeat old action and add rewards to the origial timestep
        next_state, reward, done = env.step(action, is_skip=True)
        reward_skip_step += reward

        if done:
            last_reward = batch_rewards[-1]
            last_reward += reward_skip_step
            batch_rewards[-1] = last_reward
            total_reward += reward_skip_step

    state = next_state
    step += 1

    
    if done:
        print("Step {} Score {}".format(step // skip, total_reward))

        # compute discounted rewards
        rewards = np.array(batch_rewards, dtype=np.float32)
        tmp_discounted_rewards = []
        last_r = 0
        for r in rewards[::-1]:
            last_r = r + 0.99 * last_r
            tmp_discounted_rewards.append(last_r)
        tmp_discounted_rewards.reverse()

        for r in tmp_discounted_rewards:
            batch_discounted_returns.append(r)

        # get rewards for sub-optimization reward objectives
        reward_info = env.reward_info()

        # send data to agent
        send_log_data(step // skip, total_reward)
        client_socket.close()

        break



