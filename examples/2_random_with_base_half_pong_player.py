import random

from common.half_pong_player import HalfPongPlayer


class RandomHalfPongPlayer(HalfPongPlayer):
    """
    Same as 1_random half pong player except with most code moved to a base class that will be shared with other
    examples
    """

    def get_keys_pressed(self, screen_array, feedback, terminal):
        if feedback != 0:
            print self.score()

        action_index = random.randrange(3)

        return HalfPongPlayer.action_index_to_key(action_index)


if __name__ == '__main__':
    player = RandomHalfPongPlayer()
    player.start()
