import random
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from common.half_pong_player import HalfPongPlayer


class MLPQLearningHalfPongPlayer(HalfPongPlayer):
    MEMORY_SIZE = 500000  # number of observations to remember
    STATE_FRAMES = 2

    def __init__(self):
        """
        Neural network attached to pong, no way to train it yet
        """
        super(MLPQLearningHalfPongPlayer, self).__init__(run_real_time=False, force_game_fps=6)
        self._observations = deque()

        self._input_layer, self._output_layer = self._create_network()

        init = tf.initialize_all_variables()
        self._session = tf.Session()
        self._session.run(init)

    def _create_network(self):
        input_layer = tf.placeholder("float", [self.SCREEN_WIDTH, self.SCREEN_HEIGHT])

        feed_forward_weights_1 = tf.Variable(tf.truncated_normal([self.SCREEN_WIDTH, self.SCREEN_HEIGHT], stddev=0.01))
        feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

        feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, self.ACTIONS_COUNT], stddev=0.01))
        feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[self.ACTIONS_COUNT]))

        hidden_layer = tf.nn.relu(
            tf.matmul(input_layer, feed_forward_weights_1) + feed_forward_bias_1)

        output_layer = tf.matmul(hidden_layer, feed_forward_weights_2) + feed_forward_bias_2

        return input_layer, output_layer

    def train(self):
        pass

    def get_keys_pressed(self, screen_array, reward, terminal):
        # images will be black or white
        _, binary_image = cv2.threshold(cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY), 1, 255,
                                        cv2.THRESH_BINARY)

        # first frame must be handled differently
        if self._last_state is None:
            # the _last_state will contain the image data from the last self.STATE_FRAMES frames
            self._last_state = np.stack(tuple(binary_image for _ in range(self.STATE_FRAMES)), axis=2)
            random_action = random.randrange(self.ACTIONS_COUNT)
            self._last_action = random_action
            return self.action_index_to_key(random_action)

        binary_image = np.reshape(binary_image, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT, 1))
        current_state = np.append(self._last_state[:, :, 1:], binary_image, axis=2)

        self._observations.append((self._last_state, self._last_action, reward, current_state, terminal))

        output = self._session.run(self._input_layer, feed_dict={self._output_layer: binary_image})
        action = np.argmax(output)

        self._last_state = current_state
        self._last_action = action
        return self.action_index_to_key(action)


if __name__ == '__main__':
    player = MLPQLearningHalfPongPlayer()
    player.start()
