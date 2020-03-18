import socket
from agent import Agent
import time
import numpy as np
import pickle
from collections import deque
from pickle import UnpicklingError

def compute_std(step):
    # scale from -0.5 to -1.6 in 2_000_000 timesteps
    # clip lower and higher values
    return np.clip(-1.6e-7 * step - 0.3, -1.2, 0.3)

# Agent
agent = Agent(state_size=16, action_size=8)
agent.update_actor_model()

# Trajectory Storage
horizon = 1024
batch_size = 64
epochs = 15

batch_step = 0
batch_state = []
batch_action = []
batch_old_log_probs = []
batch_discounted_rewards = []
batch_values = []
batch_advantages = []

# Log Settings

# Stop target velocity:

# If you want to test with a already trained control policy (target velocity after 10 million steps)
step = 0 
episode = 0
last_episode_scores = deque(maxlen=250)
save_frequency = 2000  # Every 2000 updates save graph

host = socket.gethostname()  # as both code is running on same pc
port = 50005  # socket server port number

socket = socket.socket()
socket.bind((host, port))

socket.listen(3)
print("Started Agent... Listening to Events")

while True:
    conn, address = socket.accept()

    while True:

        data = conn.recv(2048)

        if data:
            log_std = np.stack([np.ones((8,), dtype=np.float32) * compute_std(step)])

            message = None # sometimes this computer is to fast, wait 1 sec in case of error
            message = pickle.loads(data)

            method = message['method']

            if method == 'action':
                state = message['state']
                action, old_log_probs, value = agent.act(state, log_std, step)
                reply = [action, old_log_probs, value]
                reply_bytes = pickle.dumps(reply)
                conn.sendall(reply_bytes)

            if method == 'log':
                    ep_step, total_reward = message['data'][0], message['data'][1]
                    step += ep_step
                    batch_step += ep_step
                    episode += 1
                    last_episode_scores.append(total_reward)
                    score_std = np.array(last_episode_scores).std()


                    print("Step {} (+{}) Episode {}, Score {}".format(step, ep_step, episode, total_reward))

                    agent.save_model(episode, frequency=save_frequency)

                    # Add new data to storage
                    message = pickle.load(open('./vars/batch.p', 'rb'))
                    batch_state = batch_state + message['states']
                    batch_action = batch_action + message['actions']
                    batch_discounted_rewards = batch_discounted_rewards + message['returns']
                    batch_old_log_probs = batch_old_log_probs + message['old_log_probs']
                    batch_values = batch_values + message['values']

                    values = message['values']
                    next_values = message['next_values']
                    rewards = message['rewards']
                    gaes = agent.compute_gae(rewards, values, next_values)

                    batch_advantages = batch_advantages + gaes.flatten().tolist()

                    reward_info = message['reward_info']

                    agent.log_score(step, total_reward, score_std, episode)
                    agent.log_reward_info(step, reward_info, ep_step, episode)


                    if batch_step > horizon:

                        print("Training with collected trajectory data.")

                        # load and shuffle data for training
                        random_idx = np.random.permutation(range(int(horizon)))

                        states = np.array(batch_state, dtype=np.float32)
                        states = states[random_idx]
                        actions = np.array(batch_action, dtype=np.float32)
                        actions = actions[random_idx]
                        discounted_returns = np.array(batch_discounted_rewards, dtype=np.float32)
                        discounted_returns = discounted_returns[random_idx]
                        old_log_probs = np.array(batch_old_log_probs, dtype=np.float32)
                        old_log_probs = old_log_probs[random_idx]
                        values = np.array(batch_values, dtype=np.float32)
                        values = values[random_idx]
                        log_std = np.ones((8,), dtype=np.float32) * compute_std(step)
                        log_stds = np.vstack([log_std] * horizon)

                        advantages = np.array(batch_advantages, dtype=np.float32)
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        advantages = advantages[random_idx]

                        # compute advantage
                        # advantages = discounted_returns - values
                        # 
                        summary = None
                        for e in range(epochs):
                            for i in range(int(horizon) // batch_size):
                                start_idx = i * batch_size
                                end_idx = start_idx + batch_size
                                train_state = states[start_idx:end_idx]
                                train_actions = actions[start_idx:end_idx]
                                train_old_log_probs = old_log_probs[start_idx:end_idx]
                                train_rewards = discounted_returns[start_idx:end_idx]
                                train_advantages = advantages[start_idx:end_idx]
                                train_log_stds = log_stds[start_idx:end_idx]

                                _, summary = agent.train(states=train_state, actions=train_actions,
                                                         rewards=train_rewards, old_log_probs=train_old_log_probs,
                                                         advantages=train_advantages, log_stds=train_log_stds)


                        agent.log_summary(summary=summary, step=step)
                        agent.update_actor_model()


                        # Reset for next batch
                        batch_step = 0

                        batch_state = []
                        batch_action = []
                        batch_discounted_rewards = []
                        batch_old_log_probs = []
                        batch_values = []
                        batch_advantages = []

                    conn.close()
                    break





