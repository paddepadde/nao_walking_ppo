import tensorflow as tf
import numpy as np
import time

from collections import deque

class Agent:
    
    def __init__(self, state_size, action_size, mode='anneal', checkpoint=None):
        
        # size of input, output of neural nets
        self.state_size = state_size
        self.action_size = action_size

        # defines how log_std of action distribution is selected
        # valid ('anneal' or 'learn')
        self.mode = mode
        self.sess = tf.Session()

        # name of folder logged to tensorboard 
        # files get quite big (>100Mb) for longer runs
        log_dir = './log/'
        self.writer = tf.summary.FileWriter(log_dir)

        with tf.variable_scope('ppo'):

            # states and rewards
            self.states = tf.placeholder(tf.float32, [None, self.state_size], name='states')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')

            # actions and old log probs
            self.actions = tf.placeholder(tf.float32, [None, self.action_size], name='actions')
            self.old_log_probs = tf.placeholder(tf.float32, [None, self.action_size], name='old_log_probs')
            self.log_stds = tf.placeholder(tf.float32, [None, self.action_size], name="log_probs")

            # advantage
            self.advantage = tf.placeholder(tf.float32, [None], name="advantage")

            # actor and critic networks
            policy, self.value = self._build_model()

            # placeholder for old actor network
            old_policy = self._build_model_old()
            trainable_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ppo')

            # coefficient for value loss, deprecated
            value_factor = 1.0

            # coefficient for entropy loss
            entropy_factor = 0.02

            # clip hyperparameter
            epsilon = 0.2

            advantage = tf.reshape(self.advantage, [-1, 1])
            returns = tf.reshape(self.rewards, [-1, 1])

            with tf.variable_scope('value_loss'):
                # MSE loss for critic
                value_loss = tf.reduce_mean(tf.square(returns - self.value))
                # tried Huber loss but no measureable difference
                # value_loss = tf.losses.huber_loss(returns, self.value)
                value_loss = value_factor * value_loss

                log_value_loss = tf.summary.scalar('loss/value', value_loss)

            with tf.variable_scope('entropy_loss'):
                # entropy loss based on action distribution
                entropy_loss = tf.reduce_mean(policy.entropy())
                entropy_loss = entropy_factor * entropy_loss

                log_entropy_loss = tf.summary.scalar('loss/entropy', entropy_loss)

            with tf.variable_scope('policy_loss'):

                # compute probability values for old and new policy
                log_prob = policy.log_prob(self.actions)
                old_log_prob = old_policy.log_prob(self.actions)

                # compute ratio r
                ratio = tf.exp(log_prob - old_log_prob)

                # stop update in case kl_div gets to big
                kl_div = tf.reduce_mean(tf.distributions.kl_divergence(old_policy, policy))
                
                # term is 0 if kl_div > max_kl, 1 otherwise
                max_kl = 0.05
                allow_update = 1.0 - tf.maximum(tf.sign((1 / max_kl) * kl_div - 1), 0.0)
            
                log_kl_div = tf.summary.scalar('/kl_div', kl_div)
                log_allow_update = tf.summary.scalar('/kl_div_allow', allow_update)

                # compute surrogate terms, use minimum
                surrogate_1 = ratio * advantage
                surrogate_2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage

                surrogate = tf.minimum(surrogate_1, surrogate_2)
                policy_loss = tf.reduce_mean(surrogate) * allow_update

                log_policy_loss = tf.summary.scalar('loss/policy', policy_loss)

            self.loss = value_loss - policy_loss - entropy_loss


            self.actor_loss = -policy_loss - entropy_loss
            self.critic_loss = value_loss

            self.trainable_weights_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/policy')
            self.trainable_weights_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo/value')

            self.old_weights_actor = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ppo/policy_old') 
            self.update_old_actor = [oldp.assign(p) for p, oldp in zip(self.trainable_weights_actor, self.old_weights_actor)]

            actor_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4, epsilon=1e-5)
            self.train_actor = actor_optimizer.minimize(self.actor_loss, var_list=self.trainable_weights_actor)

            # clip critic gradients
            critic_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-5)
            self.train_critic = critic_optimizer.minimize(self.critic_loss, var_list=self.trainable_weights_critic)
            # deprecated
            gradients = tf.gradients(self.loss, trainable_weights)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, gardient_clip)
            gradients_and_weights = zip(clipped_gradients, trainable_weights)
            optimizer = tf.train.AdamOptimizer(learning_rate=3e-5, epsilon=1e-5)
            self.train_op = optimizer.apply_gradients(gradients_and_weights)

            self.log_merge = tf.summary.merge_all()

            # sample from action distributuion
            self.action = policy.sample(1)[0]
            self.log_prob = policy.log_prob(self.action)

        if checkpoint is None:
            self.sess.run(tf.global_variables_initializer())

        # save model ./model/walking__{}
        self.saver = tf.train.Saver(max_to_keep=20)

        # if checkpoint parameter is provided, load the weights from model
        if checkpoint is not None:
            self.saver.restore(self.sess, checkpoint)


    def act(self, state, log_std, step):
        """ Get prediction for action, prob. values, and value 
        function estimate from the agent. 
        Use state as input for actor- and critic networks.

        Args:
            state: the current state of the environment
            log_std: logarithm of standard derivtion of Normal distribution
            step: current step in training (for logging)
        Returns:
            out[0][0]: the predicted action vector
            out[1][0]: the log probability of the predicted actions
            out[2][0]: the value function estimate
        """
        feed_dict = {self.states: state, self.log_stds: log_std}
        out = self.sess.run([self.action, self.log_prob, self.value, self.mean, self.std], feed_dict)

        # log std to tensorboard
        log_std = np.mean(out[4][0])
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='ppo/log_std', simple_value=log_std)]), step)
        # print(out[0][0]); print(out[3][0]); print(out[4][0])
        return out[0][0], out[1][0], out[2][0]   # actions, old_log_probs, value

    def compute_gae(self, reward, value, next_value, gamma=0.99, lam=0.97):
        """ Compute estimate of advantage using generalized advantage 
        estimation.
        See: Schulman et. al, High-Dimensional Continuous Control Using 
        Generalized Advantage Estimation

        Args:
            reward: vector of rewards received in episode
            value: value function estimates received in episode
            next_value: value function estimate for subsequent state
            gamma, lam: temporal decay parameters

        Returns:
            gaes: vector with estimates of advantage
        """
        deltas = [r_t + gamma * v_next - v for r_t, v_next, v in zip(reward, next_value, value)]

        #  see ppo paper eq(11)
        gaes = np.copy(deltas)
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + gamma * lam * gaes[t + 1]
        return gaes

    def train(self, states, actions, rewards, old_log_probs, advantages, log_stds):
        """ Train actor and critic networks with values from trajectories. 

        Args:
            states: list of observed states 
            actions: list of observed actions
            rewards: list of stacked rewards which were observed in the 
            old_log_probs: list of observed probability values
            advantages: list of computed advantages (gae)
            log_stds: logarithm of standard derivtion of Normal distribution

        Returns:
            out: ignore
            summary: log data for tensorboard (losses)
        """
        feed_dict = {self.states: states, self.actions: actions, 
            self.rewards: rewards, self.old_log_probs: old_log_probs, 
            self.advantage: advantages, self.log_stds: log_stds}
        out = self.sess.run([self.actor_loss, self.critic_loss, self.log_merge, self.train_actor, self.train_critic], feed_dict)
        
        summary = out[2]
        return out, summary

    def log_score(self, step, score, score_std, episode):

        if episode % 25 != 0:
            return

        self.writer.add_summary(
            tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=score)]), step)
        self.writer.add_summary(
            tf.Summary(value=[tf.Summary.Value(tag='reward/score_std', simple_value=score_std)]), step)

    def log_summary(self, step, summary):
        self.writer.add_summary(summary, step)

    def log_reward_info(self, step, reward_info, ep_steps, episode):

        if episode % 25 != 0:
            return
            
        distance_reward = reward_info[0]
        alive_reward = reward_info[1]
        movement_reward = reward_info[2]
        clip_reward = reward_info[3]
        target_distance_reward = reward_info[4]

        norm_distance = distance_reward / ep_steps
        norm_alive = alive_reward / ep_steps
        norm_movement= movement_reward / ep_steps
        norm_clip = clip_reward / ep_steps
        norm_target_distance = target_distance_reward / ep_steps

        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='reward-info/distance', simple_value=distance_reward)]), step)
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='reward-info/alive', simple_value=alive_reward)]), step)
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='reward-info/movement', simple_value=movement_reward)]), step)
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='reward-info/clip', simple_value=clip_reward)]), step)
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='reward-info/target_distance', simple_value=target_distance_reward)]), step)

        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='reward-norm/distance', simple_value=norm_distance)]), step)
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='reward-norm/alive', simple_value=norm_alive)]), step)
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='reward-norm/movement', simple_value=norm_movement)]), step)
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='reward-norm/clip', simple_value=norm_clip)]), step)
        self.writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='reward-norm/target_distance', simple_value=norm_target_distance)]), step)

    # save current model every FREQUENCY steps
    def save_model(self, step, frequency=2000):
        if(step % frequency == 0):
            model_name = './model/walking__{}'.format(step)
            self.saver.save(self.sess, model_name)
            print("Saved model to {}".format(model_name))

    def save_session(self):
        self.saver.save(self.sess, './model/save')

    def load_session(self, reset=False):
        if not reset:
            self.saver.restore(self.sess, './model/save.ckpt')

    def update_actor_model(self):
        self.sess.run(self.update_old_actor)


    def _build_model(self):

        with tf.variable_scope('policy'):
            p_layer_1 = tf.layers.dense(self.states, units=32, activation=tf.nn.tanh)
            p_layer_2 = tf.layers.dense(p_layer_1, units=32, activation=tf.nn.tanh)

            self.mean = tf.layers.dense(p_layer_2, units=self.action_size, activation=None)
            log_std = tf.get_variable(name='log_std', shape=[1, self.action_size], initializer=tf.zeros_initializer())
            
            if self.mode == 'anneal':
                self.std = tf.exp(self.log_stds)
            else:
                self.std = tf.zeros_like(self.mean) + tf.exp(log_std)

            self.std = tf.clip_by_value(self.std, 0.15, 3.0)

            policy = tf.distributions.Normal(loc=self.mean, scale=self.std)

        with tf.variable_scope('value'):
            v_layer_1 = tf.layers.dense(self.states, units=32, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            v_layer_2 = tf.layers.dense(v_layer_1, units=32, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())

            value = tf.layers.dense(v_layer_2, units=1, activation=None)

        return policy, value

    
    def _build_model_old(self):

        with tf.variable_scope('policy_old'):

            p_layer_1 = tf.layers.dense(self.states, units=32, activation=tf.nn.tanh, trainable=False)
            p_layer_2 = tf.layers.dense(p_layer_1, units=32, activation=tf.nn.tanh, trainable=False)

            mean = tf.layers.dense(p_layer_2, units=self.action_size, activation=None, trainable=False)
            log_std = tf.get_variable(name='log_std', shape=[1, self.action_size], initializer=tf.zeros_initializer(), trainable=False)

            if self.mode == 'anneal':
                std = tf.exp(self.log_stds)
            else:
                std = tf.zeros_like(mean) + tf.exp(log_std)

            std = tf.clip_by_value(std, 0.15, 3.0)

            old_policy = tf.distributions.Normal(loc=mean, scale=std)

        return old_policy

            

