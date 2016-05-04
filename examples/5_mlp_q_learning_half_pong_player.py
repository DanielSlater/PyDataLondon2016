import os
import random
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from common.half_pong_player import HalfPongPlayer


class MLPQLearningHalfPongPlayer(HalfPongPlayer):
    MEMORY_SIZE = 500000  # number of observations to remember
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    OBSERVATION_STEPS = 500.  # time steps to observe before training
    EXPLORE_STEPS = 500000.  # frames over which to anneal epsilon
    INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
    FINAL_RANDOM_ACTION_PROB = 0.05  # final chance of an action being random
    MEMORY_SIZE = 500000  # number of observations to remember
    MINI_BATCH_SIZE = 100  # size of mini batches
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_TERMINAL_INDEX = range(5)
    LEARN_RATE = 0.0001
    SAVE_EVERY_X_STEPS = 5000
    SCREEN_WIDTH = 40
    SCREEN_HEIGHT = 40

    def __init__(self, checkpoint_path="mlp_q_learning_half_pong", playback_mode=False, verbose_logging=True):
        """
        MLP now trianing using Q-learning
        """
        self._playback_mode = playback_mode
        super(MLPQLearningHalfPongPlayer, self).__init__(run_real_time=False, force_game_fps=6)
        self.verbose_logging = verbose_logging
        self._checkpoint_path = checkpoint_path

        self._probability_of_random_action = self.INITIAL_RANDOM_ACTION_PROB
        self._observations = deque()
        self._time = 0
        self._last_action = None
        self._last_state = None

        self._input_layer, self._output_layer = self._create_network()

        self._actions = tf.placeholder("float", [None, self.ACTIONS_COUNT], name="actions")
        self._target = tf.placeholder("float", [None], name="target")

        readout_action = tf.reduce_sum(tf.mul(self._output_layer, self._actions), reduction_indices=1)

        cost = tf.reduce_mean(tf.square(self._target - readout_action))
        self._train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(cost)

        init = tf.initialize_all_variables()
        self._session = tf.Session()
        self._session.run(init)

        if not os.path.exists(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)
        self._saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self._checkpoint_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)
        elif playback_mode:
            raise Exception("Could not load checkpoints for playback")


    def _create_network(self):
        input_layer = tf.placeholder("float", [None, self.SCREEN_WIDTH * self.SCREEN_HEIGHT], name="input_layer")

        feed_forward_weights_1 = tf.Variable(
            tf.truncated_normal([self.SCREEN_WIDTH * self.SCREEN_HEIGHT, 256], stddev=0.01))
        feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

        feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, self.ACTIONS_COUNT], stddev=0.01))
        feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[self.ACTIONS_COUNT]))

        hidden_layer = tf.nn.relu(
            tf.matmul(input_layer, feed_forward_weights_1) + feed_forward_bias_1, name="hidden_activations")

        output_layer = tf.matmul(hidden_layer, feed_forward_weights_2) + feed_forward_bias_2

        return input_layer, output_layer

    def get_keys_pressed(self, screen_array, reward, terminal):
        # images will be black or white
        _, binary_image = cv2.threshold(cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY), 1, 255,
                                        cv2.THRESH_BINARY)

        binary_image = np.reshape(binary_image, (80*80,))

        # first frame must be handled differently
        if self._last_state is None:
            self._last_state = binary_image
            random_action = random.randrange(self.ACTIONS_COUNT)

            self._last_action = np.zeros([self.ACTIONS_COUNT])
            self._last_action[random_action] = 1.

            return self.action_index_to_key(random_action)

        self._observations.append((self._last_state, self._last_action, reward, binary_image, terminal))

        if len(self._observations) > self.MEMORY_SIZE:
                self._observations.popleft()

        if len(self._observations) > self.OBSERVATION_STEPS:
            self._train()
            self._time += 1

        # gradually reduce the probability of a random actionself.
        if self._probability_of_random_action > self.FINAL_RANDOM_ACTION_PROB \
                and len(self._observations) > self.OBSERVATION_STEPS:
            self._probability_of_random_action -= \
                (self.INITIAL_RANDOM_ACTION_PROB - self.FINAL_RANDOM_ACTION_PROB) / self.EXPLORE_STEPS

        print("Time: %s random_action_prob: %s reward %s scores %s" %
              (self._time, self._probability_of_random_action, reward,
               self.score()))

        action = self._choose_next_action(binary_image)
        self._last_state = binary_image

        self._last_action = np.zeros([self.ACTIONS_COUNT])
        self._last_action[action] = 1.
        return self.action_index_to_key(action)

    def _choose_next_action(self, binary_image):
        if random.random() <= self._probability_of_random_action:
            return random.randrange(self.ACTIONS_COUNT)
        else:
            # let the net choose our action
            output = self._session.run(self._input_layer, feed_dict={self._output_layer: binary_image})
            return np.argmax(output)

    def _train(self):
        # sample a mini_batch to train on
        mini_batch = random.sample(self._observations, self.MINI_BATCH_SIZE)
        # get the batch variables
        previous_states = [d[self.OBS_LAST_STATE_INDEX] for d in mini_batch]
        actions = [d[self.OBS_ACTION_INDEX] for d in mini_batch]
        rewards = [d[self.OBS_REWARD_INDEX] for d in mini_batch]
        current_states = [d[self.OBS_CURRENT_STATE_INDEX] for d in mini_batch]
        agents_expected_reward = []
        # this gives us the agents expected reward for each action we might
        agents_reward_per_action = self._session.run(self._output_layer, feed_dict={self._input_layer: current_states})
        for i in range(len(mini_batch)):
            if mini_batch[i][self.OBS_TERMINAL_INDEX]:
                # this was a terminal frame so there is no future reward...
                agents_expected_reward.append(rewards[i])
            else:
                agents_expected_reward.append(
                    rewards[i] + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

        # learn that these actions in these states lead to this reward
        self._session.run(self._train_operation, feed_dict={
            self._input_layer: previous_states,
            self._actions: actions,
            self._target: agents_expected_reward})

        # save checkpoints for later
        if self._time % self.SAVE_EVERY_X_STEPS == 0:
            self._saver.save(self._session, self._checkpoint_path + '/network', global_step=self._time)


if __name__ == '__main__':
    player = MLPQLearningHalfPongPlayer()
    player.start()
